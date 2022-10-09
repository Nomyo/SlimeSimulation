#pragma once

#include <vulkan/vulkan.h>
#include <VulkanDevice.h>
#include <glm/glm.hpp>

struct VulkanIamGuiWrapper
{
    VulkanDevice *device;

    VkImage fontImage = VK_NULL_HANDLE;
    VkImageView fontImageView = VK_NULL_HANDLE;
    VkDeviceMemory fontMemory = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;

    std::vector<VkPipelineShaderStageCreateInfo> shaders;

    int32_t vertexCount = 0;
    int32_t indexCount = 0;

    struct PushConstBlock {
        glm::vec2 scale;
        glm::vec2 translate;
    } pushConstBlock;

    // Pipeline related
    VkDescriptorPool descriptorPool;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorSet descriptorSet;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;

    // Buffers
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory = VK_NULL_HANDLE;
    void *vertexMapped = nullptr;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indexMemory = VK_NULL_HANDLE;
    void *indexMapped = nullptr;

    VkQueue queue;

    bool updated;
    bool visible;

    VulkanIamGuiWrapper();
    ~VulkanIamGuiWrapper();

    void CleanUp();
    void PrepareResources();
    void PreparePipeline(const VkRenderPass renderPass);
    bool UpdateBuffers();
    void Draw(VkCommandBuffer commandBuffer);

    bool CheckBox(const std::string& caption, bool* value);
    bool SliderFloat(const char* caption, float* value, float min, float max);
    bool SliderInt(const char* caption, int32_t* value, int32_t min, int32_t max);
    bool Header(const char *caption);
};
