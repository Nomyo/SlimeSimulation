#include <SlimeSimulation.h>
#include <VulkanCamera.h>
#include <VulkanUtils.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/gtc/matrix_transform.hpp>

#define ENABLE_VALIDATION true

#include <array>
#include <ctime>
#include <fstream>
#include <random>

SlimeSimulation::SlimeSimulation() : VulkanCore(ENABLE_VALIDATION)
{
}

SlimeSimulation::~SlimeSimulation()
{
    // Destroy fences
    for (auto& fence : m_queueCompleteFences) {
        vkDestroyFence(m_logicalDevice, fence, nullptr);
    }

    // Destroy textures
    m_textures.slime.Destroy();

    // Destroy compute
    vkFreeMemory(m_logicalDevice, m_compute.storageBuffer.memory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_compute.storageBuffer.buffer, nullptr);
    vkFreeMemory(m_logicalDevice, m_compute.uniformBuffer.memory, nullptr);
    vkDestroyBuffer(m_logicalDevice, m_compute.uniformBuffer.buffer, nullptr);

    vkDestroyDescriptorSetLayout(m_logicalDevice, m_compute.slime.descriptorSetLayout, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_compute.slime.pipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_compute.slime.pipeline, nullptr);
    vkDestroySemaphore(m_logicalDevice, m_compute.semaphore, nullptr);

    vkFreeCommandBuffers(m_logicalDevice, m_compute.commandPool, 1, &m_compute.commandBuffer);
    vkDestroyCommandPool(m_logicalDevice, m_compute.commandPool, nullptr);

    // Destroy graphics
    vkDestroyDescriptorSetLayout(m_logicalDevice, m_graphics.slime.descriptorSetLayout, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_graphics.slime.pipelineLayout, nullptr);
    vkDestroyPipeline(m_logicalDevice, m_graphics.slime.pipeline, nullptr);
    vkDestroySemaphore(m_logicalDevice, m_graphics.semaphore, nullptr);
}


void SlimeSimulation::Render()
{
    if (!m_prepared)
    {
        return;
    }

    Draw();
    UpdateUniformBuffers();
}


void SlimeSimulation::Prepare()
{
    VulkanCore::Prepare();

    m_graphics.queueFamilyIndex = m_vulkanDevice->queueFamilyIndices.graphics;
    m_compute.queueFamilyIndex = m_vulkanDevice->queueFamilyIndices.compute;

    m_textures.slime.CreateTargetTexture(m_width / 2 , m_height/ 2, VK_FORMAT_R8G8B8A8_UNORM, m_vulkanDevice, m_graphicsQueue, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_LAYOUT_GENERAL);

    SetupDescriptorPool();

    // Create storage buffer to be modified in compute shader
    PrepareStorageBuffers();

    // create compute UBO and get host accessible mapping
    PrepareUniformBuffers();

    PrepareGraphics();
    PrepareCompute();

    BuildCommandBuffers();
    m_prepared = true;
}

void SlimeSimulation::Draw()
{
    VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    // Submit graphics commands
    VkSubmitInfo computeSubmitInfo{};
    computeSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    computeSubmitInfo.commandBufferCount = 1;
    computeSubmitInfo.pCommandBuffers = &m_compute.commandBuffer;
    computeSubmitInfo.waitSemaphoreCount = 1;
    computeSubmitInfo.pWaitSemaphores = &m_graphics.semaphore;
    computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
    computeSubmitInfo.signalSemaphoreCount = 1;
    computeSubmitInfo.pSignalSemaphores = &m_compute.semaphore;
    VK_CHECK_RESULT(vkQueueSubmit(m_compute.queue, 1, &computeSubmitInfo, VK_NULL_HANDLE));
    // Acquire the next image
    // Note the cpu wait if there is image ready to be rendered in. However with 3 frames in flights and using a mail box presenting more
    // we should always have at least one image ready
    VulkanCore::PrepareFrame();

    VkPipelineStageFlags graphicsWaitStageMasks[] = { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    VkSemaphore graphicsWaitSemaphores[] = { m_compute.semaphore, m_semaphores.presentComplete };
    VkSemaphore graphicsSignalSemaphores[] = { m_graphics.semaphore, m_semaphores.renderComplete };

    // Submit graphics commands
    m_submitInfo.commandBufferCount = 1;
    m_submitInfo.pCommandBuffers = &m_drawCmdBuffers[m_currentBuffer];
    m_submitInfo.waitSemaphoreCount = 2;
    m_submitInfo.pWaitSemaphores = graphicsWaitSemaphores;
    m_submitInfo.pWaitDstStageMask = graphicsWaitStageMasks;
    m_submitInfo.signalSemaphoreCount = 2;
    m_submitInfo.pSignalSemaphores = graphicsSignalSemaphores;
    VK_CHECK_RESULT(vkQueueSubmit(m_graphicsQueue, 1, &m_submitInfo, VK_NULL_HANDLE));

    // Present the Frame to the queue
    // NOTE: Submit Frame does an vkQueueWaitIdle on the graphics Queue
    // So parallelism is actually enable only because it does not wait for present operation
    VulkanCore::SubmitFrame();
}

void SlimeSimulation::SetupGraphicsDescriptorSetLayout()
{
    VkDescriptorSetLayoutBinding slimeSamplerBinding{};
    slimeSamplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    slimeSamplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    slimeSamplerBinding.binding = 0;
    slimeSamplerBinding.descriptorCount = 1;

    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
        slimeSamplerBinding
    };

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo{};
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.pBindings = setLayoutBindings.data();
    descriptorSetLayoutCreateInfo.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
    descriptorSetLayoutCreateInfo.pNext = nullptr;
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(m_logicalDevice, &descriptorSetLayoutCreateInfo, nullptr, &m_graphics.slime.descriptorSetLayout));

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pNext = nullptr;
    pipelineLayoutCreateInfo.pSetLayouts = &m_graphics.slime.descriptorSetLayout;
    VK_CHECK_RESULT(vkCreatePipelineLayout(m_logicalDevice, &pipelineLayoutCreateInfo, nullptr, &m_graphics.slime.pipelineLayout));
}

void SlimeSimulation::PrepareGraphicsPipelines()
{
    // Slime Pipeline
    PrepareSlimePipeline();
}

void SlimeSimulation::SetupDescriptorPool()
{
    VkDescriptorPoolSize descriptorPoolUniformSize{};
    descriptorPoolUniformSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorPoolUniformSize.descriptorCount = 1;

    VkDescriptorPoolSize descriptorPoolStorageBufferSize{};
    descriptorPoolStorageBufferSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorPoolStorageBufferSize.descriptorCount = 1;

    VkDescriptorPoolSize descriptorPoolStorageImageSize{};
    descriptorPoolStorageImageSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptorPoolStorageImageSize.descriptorCount = 1;

    VkDescriptorPoolSize descriptorPoolImageSampler{};
    descriptorPoolImageSampler.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorPoolImageSampler.descriptorCount = 1;

    std::vector<VkDescriptorPoolSize> poolSizes =
    {
        descriptorPoolUniformSize,
        descriptorPoolStorageBufferSize,
        descriptorPoolStorageImageSize,
        descriptorPoolImageSampler
    };

    VkDescriptorPoolCreateInfo descriptorPoolInfo{};
    descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    descriptorPoolInfo.pPoolSizes = poolSizes.data();
    descriptorPoolInfo.maxSets = 5;

    VK_CHECK_RESULT(vkCreateDescriptorPool(m_logicalDevice, &descriptorPoolInfo, nullptr, &m_descriptorPool));
}

void SlimeSimulation::SetupGraphicsDescriptorSet()
{
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool = m_descriptorPool;
    descriptorSetAllocateInfo.pSetLayouts = &m_graphics.slime.descriptorSetLayout;
    descriptorSetAllocateInfo.descriptorSetCount = 1;

    VK_CHECK_RESULT(vkAllocateDescriptorSets(m_logicalDevice, &descriptorSetAllocateInfo, &m_graphics.slime.descriptorSet));

    VkWriteDescriptorSet writeSamplerDescriptorSet{};
    writeSamplerDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeSamplerDescriptorSet.dstSet = m_graphics.slime.descriptorSet;
    writeSamplerDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writeSamplerDescriptorSet.dstBinding = 0;
    writeSamplerDescriptorSet.pImageInfo = &m_textures.slime.m_descriptor;
    writeSamplerDescriptorSet.descriptorCount = 1;

    std::vector<VkWriteDescriptorSet> writeDescriptorSets
    {
        writeSamplerDescriptorSet
    };

    vkUpdateDescriptorSets(m_logicalDevice, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
}

void SlimeSimulation::PrepareStorageBuffers()
{
    std::default_random_engine rndEngine((unsigned)time(nullptr));
    std::uniform_real_distribution<float> rndDistWidth(0, (float)m_textures.slime.m_width);
    std::uniform_real_distribution<float> rndDistHeight(0, (float)m_textures.slime.m_height);

    std::uniform_real_distribution<float> rndDistNorm(-1.f, 1.f);

    // Initial agents positions
    std::vector<SlimeAgent> agentBuffer(AGENT_COUNT);
    for (auto& agent : agentBuffer) {
        agent.position = glm::vec2(rndDistWidth(rndEngine), rndDistHeight(rndEngine));
        agent.direction = normalize(glm::vec2(rndDistNorm(rndEngine), rndDistNorm(rndEngine)));
    }
    VkDeviceSize storageBufferSize = agentBuffer.size() * sizeof(SlimeAgent);

    // Staging
    // SSBO won't be changed on the host after upload so copy to device local memory
    BufferWrapper stagingBuffer;

    m_vulkanDevice->CreateBuffer(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &stagingBuffer.buffer,
        &stagingBuffer.memory,
        storageBufferSize,
        agentBuffer.data());

    // FIXME: verify if host_visible useful or not
    m_vulkanDevice->CreateBuffer(
        // The SSBO will be used as a storage buffer for the compute pipeline and as a vertex buffer in the graphics pipeline
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        &m_compute.storageBuffer.buffer,
        &m_compute.storageBuffer.memory,
        storageBufferSize);

    // Copy from staging buffer to storage buffer
    VkCommandBuffer copyCmd = m_vulkanDevice->CreateCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    VkBufferCopy copyRegion = {};
    copyRegion.size = storageBufferSize;
    vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, m_compute.storageBuffer.buffer, 1, &copyRegion);

    // Set descriptor
    m_compute.storageBuffer.descriptor.buffer = m_compute.storageBuffer.buffer;
    m_compute.storageBuffer.descriptor.offset = 0;
    m_compute.storageBuffer.descriptor.range = VK_WHOLE_SIZE;

    m_vulkanDevice->FlushCommandBuffer(copyCmd, m_graphicsQueue, true);

    // Cleanup
    vkDestroyBuffer(m_logicalDevice, stagingBuffer.buffer, nullptr);
    vkFreeMemory(m_logicalDevice, stagingBuffer.memory, nullptr);
}

void SlimeSimulation::PrepareUniformBuffers()
{
    // Compute UBO
    VK_CHECK_RESULT(m_vulkanDevice->CreateBuffer(
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &m_compute.uniformBuffer.buffer,
        &m_compute.uniformBuffer.memory,
        sizeof(m_compute.ubo)));

    // Map so that further update will just have to write into the buffer
    vkMapMemory(m_logicalDevice, m_compute.uniformBuffer.memory, 0, sizeof(m_compute.ubo), 0, &m_compute.uniformBuffer.mapped);

    // Set descriptor
    m_compute.uniformBuffer.descriptor.buffer = m_compute.uniformBuffer.buffer;
    m_compute.uniformBuffer.descriptor.offset = 0;
    m_compute.uniformBuffer.descriptor.range = sizeof(m_compute.ubo);

    m_compute.ubo.spaceWidth = m_textures.slime.m_width;
    m_compute.ubo.spaceHeight = m_textures.slime.m_height;
    m_compute.ubo.agentCount = AGENT_COUNT;

    UpdateUniformBuffers();
}

void SlimeSimulation::UpdateUniformBuffers()
{
    m_compute.ubo.elapsedTime = m_frameTimer;
    memcpy(m_compute.uniformBuffer.mapped, &m_compute.ubo, sizeof(m_compute.ubo));
}

void SlimeSimulation::PrepareSlimePipeline()
{
    SetupGraphicsDescriptorSetLayout();

    // ParticlePipeline
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCreateInfo{};
    inputAssemblyStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssemblyStateCreateInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssemblyStateCreateInfo.flags = 0;
    inputAssemblyStateCreateInfo.primitiveRestartEnable = VK_FALSE;

    VkPipelineVertexInputStateCreateInfo emptyInputState{};
    emptyInputState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    emptyInputState.vertexAttributeDescriptionCount = 0;
    emptyInputState.pVertexAttributeDescriptions = nullptr;
    emptyInputState.vertexBindingDescriptionCount = 0;
    emptyInputState.pVertexBindingDescriptions = nullptr;

    VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo{};
    rasterizationStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizationStateCreateInfo.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizationStateCreateInfo.cullMode = VK_CULL_MODE_NONE;
    rasterizationStateCreateInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizationStateCreateInfo.flags = 0;
    rasterizationStateCreateInfo.depthClampEnable = VK_FALSE;
    rasterizationStateCreateInfo.lineWidth = 1.0f;

    VkPipelineColorBlendAttachmentState colorBlendAttachmentState{};
    colorBlendAttachmentState.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachmentState.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlendStateCreateInfo{};
    colorBlendStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlendStateCreateInfo.attachmentCount = 1;
    colorBlendStateCreateInfo.pAttachments = &colorBlendAttachmentState;

    VkPipelineViewportStateCreateInfo viewportStateCreateInfo{};
    viewportStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportStateCreateInfo.viewportCount = 1;
    viewportStateCreateInfo.scissorCount = 1;
    viewportStateCreateInfo.flags = 0;

    VkPipelineMultisampleStateCreateInfo multisampleStateCreateInfo{};
    multisampleStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampleStateCreateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampleStateCreateInfo.flags = 0;

    std::vector<VkDynamicState> dynamicStateEnables = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.pDynamicStates = dynamicStateEnables.data();
    dynamicState.dynamicStateCount = (uint32_t)dynamicStateEnables.size();
    dynamicState.flags = 0;

    // Rendering pipeline
    // Load shaders
    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

    shaderStages[0] = LoadShader(m_logicalDevice, "../../shaders/slime.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = LoadShader(m_logicalDevice, "../../shaders/slime.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

    VkGraphicsPipelineCreateInfo pipelineCreateInfo{};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.layout = m_graphics.slime.pipelineLayout;
    pipelineCreateInfo.renderPass = m_renderPass;
    pipelineCreateInfo.flags = 0;
    pipelineCreateInfo.basePipelineIndex = -1;
    pipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;

    pipelineCreateInfo.pVertexInputState = &emptyInputState;
    pipelineCreateInfo.pInputAssemblyState = &inputAssemblyStateCreateInfo;
    pipelineCreateInfo.pRasterizationState = &rasterizationStateCreateInfo;
    pipelineCreateInfo.pColorBlendState = &colorBlendStateCreateInfo;
    pipelineCreateInfo.pMultisampleState = &multisampleStateCreateInfo;
    pipelineCreateInfo.pViewportState = &viewportStateCreateInfo;
    pipelineCreateInfo.pDepthStencilState = nullptr;
    pipelineCreateInfo.pDynamicState = &dynamicState;
    pipelineCreateInfo.stageCount = (uint32_t)shaderStages.size();
    pipelineCreateInfo.pStages = shaderStages.data();

    VK_CHECK_RESULT(vkCreateGraphicsPipelines(m_logicalDevice, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &m_graphics.slime.pipeline));

    SetupGraphicsDescriptorSet();
}

void SlimeSimulation::PrepareGraphics()
{
    PrepareGraphicsPipelines();

    // Semaphore for compute & graphics sync
    VkSemaphoreCreateInfo semaphoreCreateInfo{};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VK_CHECK_RESULT(vkCreateSemaphore(m_logicalDevice, &semaphoreCreateInfo, nullptr, &m_graphics.semaphore));

    // Signal the semaphore
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &m_graphics.semaphore;
    VK_CHECK_RESULT(vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK_RESULT(vkQueueWaitIdle(m_graphicsQueue));

    // Fences (Used to check draw command buffer completion)
    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    // Create in signaled state so we don't wait on first render of each command buffer
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    m_queueCompleteFences.resize(m_drawCmdBuffers.size());
    for (auto& fence : m_queueCompleteFences)
    {
        VK_CHECK_RESULT(vkCreateFence(m_logicalDevice, &fenceCreateInfo, nullptr, &fence));
    }
}

void SlimeSimulation::PrepareCompute()
{
    // Create a compute capable device queue
    // The VulkanDevice::createLogicalDevice functions finds a compute capable queue and prefers queue families that only support compute
    // Depending on the implementation this may result in different queue family indices for graphics and computes,
    // requiring proper synchronization (see the memory and pipeline barriers)
    vkGetDeviceQueue(m_logicalDevice, m_compute.queueFamilyIndex, 0, &m_compute.queue);

    // Create compute pipeline
    // Compute pipelines are created separate from graphics pipelines even if they use the same queue (family index)
    VkDescriptorSetLayoutBinding slimeSSBOBinding{};
    slimeSSBOBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    slimeSSBOBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    slimeSSBOBinding.binding = 0;
    slimeSSBOBinding.descriptorCount = 1;

    VkDescriptorSetLayoutBinding slimeUBOBinding{};
    slimeUBOBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    slimeUBOBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    slimeUBOBinding.binding = 1;
    slimeUBOBinding.descriptorCount = 1;

    VkDescriptorSetLayoutBinding imageBinding{};
    imageBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    imageBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    imageBinding.binding = 2;
    imageBinding.descriptorCount = 1;

    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
        slimeSSBOBinding,
        slimeUBOBinding,
        imageBinding
    };

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo{};
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.pBindings = setLayoutBindings.data();
    descriptorSetLayoutCreateInfo.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
    descriptorSetLayoutCreateInfo.pNext = nullptr;
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(m_logicalDevice, &descriptorSetLayoutCreateInfo, nullptr, &m_compute.slime.descriptorSetLayout));

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pNext = nullptr;
    pipelineLayoutCreateInfo.pSetLayouts = &m_compute.slime.descriptorSetLayout;
    VK_CHECK_RESULT(vkCreatePipelineLayout(m_logicalDevice, &pipelineLayoutCreateInfo, nullptr, &m_compute.slime.pipelineLayout));


    // Write descriptor sets
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool = m_descriptorPool;
    descriptorSetAllocateInfo.pSetLayouts = &m_compute.slime.descriptorSetLayout;
    descriptorSetAllocateInfo.descriptorSetCount = 1;

    VK_CHECK_RESULT(vkAllocateDescriptorSets(m_logicalDevice, &descriptorSetAllocateInfo, &m_compute.slime.descriptorSet));

    VkWriteDescriptorSet slimeSSBODescriptorSet{};
    slimeSSBODescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    slimeSSBODescriptorSet.dstSet = m_compute.slime.descriptorSet;
    slimeSSBODescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    slimeSSBODescriptorSet.dstBinding = 0;
    slimeSSBODescriptorSet.pBufferInfo = &m_compute.storageBuffer.descriptor;
    slimeSSBODescriptorSet.descriptorCount = 1;

    VkWriteDescriptorSet slimeUBODescriptorSet{};
    slimeUBODescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    slimeUBODescriptorSet.dstSet = m_compute.slime.descriptorSet;
    slimeUBODescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    slimeUBODescriptorSet.dstBinding = 1;
    slimeUBODescriptorSet.pBufferInfo = &m_compute.uniformBuffer.descriptor;
    slimeUBODescriptorSet.descriptorCount = 1;

    VkWriteDescriptorSet imageDescriptorSet{};
    imageDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    imageDescriptorSet.dstSet = m_compute.slime.descriptorSet;
    imageDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    imageDescriptorSet.dstBinding = 2;
    imageDescriptorSet.pImageInfo = &m_textures.slime.m_descriptor;
    imageDescriptorSet.descriptorCount = 1;

    std::vector<VkWriteDescriptorSet> writeDescriptorSets
    {
        slimeSSBODescriptorSet,
        slimeUBODescriptorSet,
        imageDescriptorSet
    };
    vkUpdateDescriptorSets(m_logicalDevice, (uint32_t)writeDescriptorSets.size(), writeDescriptorSets.data(), 0, nullptr);

    VkComputePipelineCreateInfo computePipelineCreateInfo{};
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.layout = m_compute.slime.pipelineLayout;
    computePipelineCreateInfo.flags = 0;
    computePipelineCreateInfo.stage = LoadShader(m_logicalDevice, "../../shaders/simulation.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);
    VK_CHECK_RESULT(vkCreateComputePipelines(m_logicalDevice, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &m_compute.slime.pipeline));

    VkCommandPoolCreateInfo computeCommandPoolCreateInfo{};
    computeCommandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    computeCommandPoolCreateInfo.queueFamilyIndex = m_compute.queueFamilyIndex;
    computeCommandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK_RESULT(vkCreateCommandPool(m_logicalDevice, &computeCommandPoolCreateInfo, nullptr, &m_compute.commandPool));

    // Create a command buffer for compute operations
    m_compute.commandBuffer = m_vulkanDevice->CreateCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, m_compute.commandPool);

    // Semaphore for compute & graphics sync
    VkSemaphoreCreateInfo semaphoreCreateInfo{};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VK_CHECK_RESULT(vkCreateSemaphore(m_logicalDevice, &semaphoreCreateInfo, nullptr, &m_compute.semaphore));

    // Build a single command buffer containing the compute dispatch commands
    BuildComputeCommandBuffer();
}

void SlimeSimulation::BuildCommandBuffers()
{
    VkCommandBufferBeginInfo cmdBufInfo{};
    cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    VkClearValue clearValues = { {m_defaultClearColor} };

    VkRenderPassBeginInfo renderPassBeginInfo{};
    renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassBeginInfo.renderPass = m_renderPass;
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = m_width;
    renderPassBeginInfo.renderArea.extent.height = m_height;
    renderPassBeginInfo.clearValueCount = 1;
    renderPassBeginInfo.pClearValues = &clearValues;

    for (int32_t i = 0; i < m_drawCmdBuffers.size(); ++i)
    {
        // Set target frame buffer
        renderPassBeginInfo.framebuffer = m_frameBuffers[i];

        VK_CHECK_RESULT(vkBeginCommandBuffer(m_drawCmdBuffers[i], &cmdBufInfo));

        // Image memory barrier to make sure that compute shader writes are finished before sampling from the texture
        VkImageMemoryBarrier imageMemoryBarrier = {};
        imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        // We won't be changing the layout of the image
        imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageMemoryBarrier.image = m_textures.slime.m_image;
        imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vkCmdPipelineBarrier(
            m_drawCmdBuffers[i],
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &imageMemoryBarrier);
        vkCmdBeginRenderPass(m_drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)m_width;
        viewport.height = (float)m_height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(m_drawCmdBuffers[i], 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.extent.width = m_width;
        scissor.extent.height = m_height;
        scissor.offset.x = 0;
        scissor.offset.y = 0;

        vkCmdSetScissor(m_drawCmdBuffers[i], 0, 1, &scissor);

        VkDeviceSize offsets[1] = { 0 };
        vkCmdBindPipeline(m_drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphics.slime.pipeline);
        vkCmdBindDescriptorSets(m_drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphics.slime.pipelineLayout, 0, 1, &m_graphics.slime.descriptorSet, 0, nullptr);
        vkCmdDraw(m_drawCmdBuffers[i], 3, 1, 0, 0);

        DrawUI(m_drawCmdBuffers[i]);

        vkCmdEndRenderPass(m_drawCmdBuffers[i]);

        VK_CHECK_RESULT(vkEndCommandBuffer(m_drawCmdBuffers[i]));
    }
}

void SlimeSimulation::BuildComputeCommandBuffer()
{
    VkCommandBufferBeginInfo cmdBufInfo{};
    cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    VK_CHECK_RESULT(vkBeginCommandBuffer(m_compute.commandBuffer, &cmdBufInfo));

    // Dispatch the compute job
    vkCmdBindPipeline(m_compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_compute.slime.pipeline);
    vkCmdBindDescriptorSets(m_compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_compute.slime.pipelineLayout, 0, 1, &m_compute.slime.descriptorSet, 0, 0);
    vkCmdDispatch(m_compute.commandBuffer, AGENT_COUNT, 1, 1);

    vkEndCommandBuffer(m_compute.commandBuffer);
}

void SlimeSimulation::OnUpdateUIOverlay(VulkanIamGuiWrapper *uiWrapper)
{
    //uiWrapper->CheckBox("Attach attractor to cursor", &m_attractorMouse);
}

void SlimeSimulation::OnViewChanged()
{
}
