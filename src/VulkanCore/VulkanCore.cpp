#include <VulkanCore.h>
#include <VulkanUtils.h>

#include <imgui.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>

namespace {

    void framebufferResizeCallback(GLFWwindow* window, int width, int height)
    {
        auto app = reinterpret_cast<VulkanCore*>(glfwGetWindowUserPointer(window));
        app->WindowResize();
    }

    void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
    {
        auto app = reinterpret_cast<VulkanCore*>(glfwGetWindowUserPointer(window));
        app->MouseButtonCallback(window, button, action, mods);
    }

    void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
    {
        auto app = reinterpret_cast<VulkanCore*>(glfwGetWindowUserPointer(window));
        app->KeyCallback(window, key, scancode, action, mods);
    }

    void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos)
    {
        auto app = reinterpret_cast<VulkanCore*>(glfwGetWindowUserPointer(window));
        app->CursorPositionCallback(window, xpos, ypos);
    }

} // anomymous

VulkanCore::VulkanCore(bool enableValidation)
    : m_validation(enableValidation)
    , m_applicationName("Vulkan Application")
{
}

VulkanCore::~VulkanCore()
{
    CleanUp();
}

VkResult VulkanCore::CreateInstance(bool enableValidation)
{
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = m_applicationName.c_str();
    appInfo.pEngineName = m_applicationName.c_str();
    appInfo.apiVersion = VK_API_VERSION_1_3;

    std::vector<const char*> instanceExtensions = { VK_KHR_SURFACE_EXTENSION_NAME };

    // Get extensions supported by the instance and store for later use
    uint32_t extCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
    if (extCount > 0)
    {
        std::vector<VkExtensionProperties> extensions(extCount);
        if (vkEnumerateInstanceExtensionProperties(nullptr, &extCount, &extensions.front()) == VK_SUCCESS)
        {
            for (VkExtensionProperties extension : extensions)
            {
                m_supportedInstanceExtensions.push_back(extension.extensionName);
            }
        }
    }

    VkInstanceCreateInfo instanceCreateInfo = {};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pNext = NULL;
    instanceCreateInfo.pApplicationInfo = &appInfo;

    // Surface extensions
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    for (auto i = 0u; i < glfwExtensionCount; ++i)
    {
        auto str = glfwExtensions[i];
        m_enabledInstanceExtensions.push_back(str);
    }

    instanceCreateInfo.enabledExtensionCount = glfwExtensionCount;
    instanceCreateInfo.ppEnabledExtensionNames = glfwExtensions;

    // Enabled requested instance extensions
    if (m_enabledInstanceExtensions.size() > 0)
    {
        for (const char* enabledExtension : m_enabledInstanceExtensions)
        {
            // Output message if requested extension is not available
            if (std::find(m_supportedInstanceExtensions.begin(), m_supportedInstanceExtensions.end(), enabledExtension) == m_supportedInstanceExtensions.end())
            {
                std::cerr << "Enabled instance extension \"" << enabledExtension << "\" is not present at instance level\n";
            }
            instanceExtensions.push_back(enabledExtension);
        }
    }

    if (instanceExtensions.size() > 0)
    {
        if (enableValidation)
        {
            instanceExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
            instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
        instanceCreateInfo.ppEnabledExtensionNames = instanceExtensions.data();
    }

    if (enableValidation)
    {
        // The VK_LAYER_KHRONOS_validation contains all current validation functionality.
        const char* validationLayerName = "VK_LAYER_KHRONOS_validation";

        // Check if this layer is available at instance level
        uint32_t instanceLayerCount;
        vkEnumerateInstanceLayerProperties(&instanceLayerCount, nullptr);
        std::vector<VkLayerProperties> instanceLayerProperties(instanceLayerCount);
        vkEnumerateInstanceLayerProperties(&instanceLayerCount, instanceLayerProperties.data());

        bool validationLayerPresent = false;
        for (VkLayerProperties layer : instanceLayerProperties) {
            if (strcmp(layer.layerName, validationLayerName) == 0) {
                validationLayerPresent = true;
                break;
            }
        }
        if (validationLayerPresent) {
            instanceCreateInfo.ppEnabledLayerNames = &validationLayerName;
            instanceCreateInfo.enabledLayerCount = 1;
        }
        else {
            std::cerr << "Validation layer VK_LAYER_KHRONOS_validation not present, validation is disabled";
        }
    }
    return vkCreateInstance(&instanceCreateInfo, nullptr, &m_instance);
}

void VulkanCore::InitVulkan()
{
    // Vulkan instance
    VkResult err;
    err = CreateInstance(m_validation);
    if (err) {
        throw("Could not create Vulkan instance : \n" + Utils::errorString(err), err);
    }

    // Physical device
    uint32_t gpuCount = 0;
    // Get number of available physical devices
    VK_CHECK_RESULT(vkEnumeratePhysicalDevices(m_instance, &gpuCount, nullptr));
    if (gpuCount == 0) {
        throw("No device with Vulkan support found", -1);
    }
    // Enumerate devices
    std::vector<VkPhysicalDevice> physicalDevices(gpuCount);
    err = vkEnumeratePhysicalDevices(m_instance, &gpuCount, physicalDevices.data());
    if (err) {
        throw("Could not enumerate physical devices : \n" + Utils::errorString(err), err);
    }

    // GPU selection: defaulted to the first device for now
    m_physicalDevice = physicalDevices[0];

    // Store properties (including limits), features and memory properties of the physical device
    vkGetPhysicalDeviceProperties(m_physicalDevice, &m_deviceProperties);
    vkGetPhysicalDeviceFeatures(m_physicalDevice, &m_deviceFeatures);
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &m_deviceMemoryProperties);

#if 0
    // Derived examples can override this to set actual features (based on above readings) to enable for logical device creation
    getEnabledFeatures();
#endif

    // Vulkan device creation that encapsulate functions related to a device
    m_vulkanDevice = new VulkanDevice(m_physicalDevice);
    VkResult res = m_vulkanDevice->CreateLogicalDevice(m_enabledFeatures, m_enabledDeviceExtensions);
    if (res != VK_SUCCESS) {
        throw("Could not create Vulkan device: \n" + Utils::errorString(res), res);
    }

    m_logicalDevice = m_vulkanDevice->logicalDevice;

    // Pass the necessary handle to the swapChain wrapper
    m_swapChain.Init(m_instance, m_physicalDevice, m_logicalDevice);

    // Get a graphics queue from the device
    vkGetDeviceQueue(m_logicalDevice, m_vulkanDevice->queueFamilyIndices.graphics, 0, &m_graphicsQueue);

    // Create synchronisation objects
    VkSemaphoreCreateInfo semaphoreCreateInfo{};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    // Create a semaphore used to synchronize image presentation
    // Ensures that the image is displayed before we start submitting new commands to the queue
    VK_CHECK_RESULT(vkCreateSemaphore(m_logicalDevice, &semaphoreCreateInfo, nullptr, &m_semaphores.presentComplete));
    // Create a semaphore used to synchronize command submission
    // Ensures that the image is not presented until all commands have been submitted and executed
    VK_CHECK_RESULT(vkCreateSemaphore(m_logicalDevice, &semaphoreCreateInfo, nullptr, &m_semaphores.renderComplete));

    // Set up submit info structure
    // Semaphores will stay the same during application lifetime
    // Command buffer submission info is set by each example
    m_submitInfo = VkSubmitInfo{};
    m_submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    m_submitInfo.pWaitDstStageMask = &m_submitPipelineStages;
    m_submitInfo.waitSemaphoreCount = 1;
    m_submitInfo.pWaitSemaphores = &m_semaphores.presentComplete;
    m_submitInfo.signalSemaphoreCount = 1;
    m_submitInfo.pSignalSemaphores = &m_semaphores.renderComplete;
}

void VulkanCore::MouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
    {
        m_mouseButtons.right = true;
    }
    else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE)
    {
        m_mouseButtons.right = false;
    }
    else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    {
        m_mouseButtons.left = true;
    }
    else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
    {
        m_mouseButtons.left = false;
    }
}

void VulkanCore::KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    // PRESS
    if ((key == GLFW_KEY_UP || key == GLFW_KEY_W) && action == GLFW_PRESS)
    {
        m_camera.keys.up = true;
    }
    else if ((key == GLFW_KEY_DOWN || key == GLFW_KEY_S) && action == GLFW_PRESS)
    {
        m_camera.keys.down = true;
    }
    else if ((key == GLFW_KEY_LEFT || key == GLFW_KEY_A) && action == GLFW_PRESS)
    {
        m_camera.keys.left = true;
    }
    else if ((key == GLFW_KEY_RIGHT || key == GLFW_KEY_D) && action == GLFW_PRESS)
    {
        m_camera.keys.right = true;
    }
    else if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    {
        m_camera.keys.space = true;
    }
    else if (key == GLFW_KEY_LEFT_CONTROL && action == GLFW_PRESS)
    {
        m_camera.keys.ctrl = true;
    }
    // RELEASE
    else if ((key == GLFW_KEY_UP || key == GLFW_KEY_W) && action == GLFW_RELEASE)
    {
        m_camera.keys.up = false;
    }
    else if ((key == GLFW_KEY_DOWN || key == GLFW_KEY_S) && action == GLFW_RELEASE)
    {
        m_camera.keys.down = false;
    }
    else if ((key == GLFW_KEY_LEFT || key == GLFW_KEY_A) && action == GLFW_RELEASE)
    {
        m_camera.keys.left = false;
    }
    else if ((key == GLFW_KEY_RIGHT || key == GLFW_KEY_D) && action == GLFW_RELEASE)
    {
        m_camera.keys.right = false;
    }
    else if (key == GLFW_KEY_SPACE && action == GLFW_RELEASE)
    {
        m_camera.keys.space = false;
    }
    else if (key == GLFW_KEY_LEFT_CONTROL && action == GLFW_RELEASE)
    {
        m_camera.keys.ctrl = false;
    }
}

void VulkanCore::CursorPositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    double dx = m_mousePosX - xpos;
    double dy = m_mousePosY - ypos;

    if (m_mouseButtons.left) {
        m_camera.Rotate(glm::vec3(dy * m_camera.GetRotationSpeed(), -dx * m_camera.GetRotationSpeed(), 0.0f));
        m_viewUpdated = true;
    }
}

void VulkanCore::SetupWindow()
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    m_pWindow = glfwCreateWindow(m_width, m_height, "Physarum transport networks", nullptr, nullptr);
    glfwSetWindowUserPointer(m_pWindow, this);

    glfwSetMouseButtonCallback(m_pWindow, mouseButtonCallback);
    glfwSetKeyCallback(m_pWindow, keyCallback);
    glfwSetCursorPosCallback(m_pWindow, cursorPositionCallback);
    glfwSetFramebufferSizeCallback(m_pWindow, framebufferResizeCallback);
}

void VulkanCore::Prepare()
{
    // Init swapchain surface
    m_swapChain.InitSurface(m_pWindow);

    // Create command buffer pool on Graphics Queue
    VkCommandPoolCreateInfo cmdPoolInfo = {};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.queueFamilyIndex = m_swapChain.GetQueueIndex();
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK_RESULT(vkCreateCommandPool(m_logicalDevice, &cmdPoolInfo, nullptr, &m_cmdPool));

    // Setup the swapchain
    m_swapChain.Create(&m_width, &m_height);

    // Create renderPass
    SetupRenderPass();

    CreateCommandBuffers();

    CreateSynchronizationPrimitives();

    SetupFrameBuffer();

    m_ui.device = m_vulkanDevice;
    m_ui.queue = m_graphicsQueue;
    m_ui.shaders = {
        LoadShader(m_logicalDevice, "../../shaders/ui.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
        LoadShader(m_logicalDevice, "../../shaders/ui.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT),
    };
    m_ui.PrepareResources();
    m_ui.PreparePipeline(m_renderPass);
}

VkPipelineShaderStageCreateInfo VulkanCore::LoadShader(VkDevice device, const std::string& filepath, VkShaderStageFlagBits stage)
{
    auto ucode = Utils::ReadFile(filepath);
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = ucode.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(ucode.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
    }

    m_shaderModules.push_back(shaderModule);

    VkPipelineShaderStageCreateInfo shaderStageInfo{};
    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage = stage;
    shaderStageInfo.module = shaderModule;
    shaderStageInfo.pName = "main";

    return shaderStageInfo;
}

void VulkanCore::CleanUp()
{
    // Clean up Vulkan resources
    m_swapChain.CleanUp();
    vkDestroyDescriptorPool(m_logicalDevice, m_descriptorPool, nullptr);

    vkFreeCommandBuffers(m_logicalDevice, m_cmdPool, static_cast<uint32_t>(m_drawCmdBuffers.size()), m_drawCmdBuffers.data());
    vkDestroyRenderPass(m_logicalDevice, m_renderPass, nullptr);
    for (uint32_t i = 0; i < m_frameBuffers.size(); i++)
    {
        vkDestroyFramebuffer(m_logicalDevice, m_frameBuffers[i], nullptr);
    }

    vkDestroyCommandPool(m_logicalDevice, m_cmdPool, nullptr);

    vkDestroySemaphore(m_logicalDevice, m_semaphores.presentComplete, nullptr);
    vkDestroySemaphore(m_logicalDevice, m_semaphores.renderComplete, nullptr);
    for (auto& fence : m_waitFences) {
        vkDestroyFence(m_logicalDevice, fence, nullptr);
    }

    m_ui.CleanUp();

    for (auto& shaderModule : m_shaderModules)
    {
        vkDestroyShaderModule(m_logicalDevice, shaderModule, nullptr);
    }

    delete m_vulkanDevice;
    vkDestroyInstance(m_instance, nullptr);
}

void VulkanCore::RenderLoop()
{
    m_lastTimestamp = std::chrono::high_resolution_clock::now();
    m_tPrevEnd = m_lastTimestamp;

    while (!glfwWindowShouldClose(m_pWindow)) {
        glfwPollEvents();
        if (m_prepared) {
            NextFrame();
        }
    }

    // Flush device to make sure all resources can be freed
    if (m_logicalDevice != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_logicalDevice);
    }
}

void VulkanCore::OnViewChanged()
{
}

void VulkanCore::NextFrame()
{
    glfwGetCursorPos(m_pWindow, &m_mousePosX, &m_mousePosY);

    // TODO:Handler view updates camera, etc.
    if (m_viewUpdated) {
        // Update uniform buffers in the main application
        OnViewChanged();
        m_viewUpdated = false;
    }

    auto tStart = std::chrono::high_resolution_clock::now();

    Render();
    m_frameCounter++;

    auto tEnd = std::chrono::high_resolution_clock::now();
    auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

    m_frameTimer = (float)tDiff / 1000.0f;

    m_camera.Update(m_frameTimer);
    if (m_camera.HasViewChanged()) {
        m_viewUpdated = true;
    }

    float fpsTimer = (float)(std::chrono::duration<double, std::milli>(tEnd - m_lastTimestamp).count());
    if (fpsTimer > 1000.0f)
    {
        m_lastFPS = static_cast<uint32_t>((float)m_frameCounter * (1000.0f / fpsTimer));
        m_frameCounter = 0;
        m_lastTimestamp = tEnd;
    }
    m_tPrevEnd = tEnd;

    UpdateUI();
}

void VulkanCore::SetupRenderPass()
{
    VkAttachmentDescription attachments = {};
    // Color attachment
    attachments.format = m_swapChain.GetColorFormat();
    attachments.samples = VK_SAMPLE_COUNT_1_BIT;
    attachments.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorReference = {};
    colorReference.attachment = 0;
    colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpassDescription = {};
    subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDescription.colorAttachmentCount = 1;
    subpassDescription.pColorAttachments = &colorReference;
    subpassDescription.inputAttachmentCount = 0;
    subpassDescription.pInputAttachments = nullptr;
    subpassDescription.preserveAttachmentCount = 0;
    subpassDescription.pPreserveAttachments = nullptr;
    subpassDescription.pResolveAttachments = nullptr;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &attachments;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpassDescription;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    VK_CHECK_RESULT(vkCreateRenderPass(m_logicalDevice, &renderPassInfo, nullptr, &m_renderPass));
}

void VulkanCore::SetupFrameBuffer()
{
    VkImageView attachments;

    VkFramebufferCreateInfo frameBufferCreateInfo = {};
    frameBufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    frameBufferCreateInfo.pNext = NULL;
    frameBufferCreateInfo.renderPass = m_renderPass;
    frameBufferCreateInfo.attachmentCount = 1;
    frameBufferCreateInfo.pAttachments = &attachments;
    frameBufferCreateInfo.width = m_width;
    frameBufferCreateInfo.height = m_height;
    frameBufferCreateInfo.layers = 1;

    // Create frame buffers for every swap chain image
    m_frameBuffers.resize(m_swapChain.GetImageCount());
    for (uint32_t i = 0; i < m_frameBuffers.size(); i++)
    {
        attachments = m_swapChain.GetImageView(i);
        VK_CHECK_RESULT(vkCreateFramebuffer(m_logicalDevice, &frameBufferCreateInfo, nullptr, &m_frameBuffers[i]));
    }
}

void VulkanCore::CreateCommandBuffers()
{
    // Create one command buffer for each swap chain image and reuse for rendering
    m_drawCmdBuffers.resize(m_swapChain.GetImageCount());

    VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = m_cmdPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = static_cast<uint32_t>(m_drawCmdBuffers.size());

    VK_CHECK_RESULT(vkAllocateCommandBuffers(m_logicalDevice, &commandBufferAllocateInfo, m_drawCmdBuffers.data()));
}

void VulkanCore::CreateSynchronizationPrimitives()
{
    // Wait fences to sync command buffer access
    VkFenceCreateInfo fenceCreateInfo{};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    m_waitFences.resize(m_drawCmdBuffers.size());
    for (auto& fence : m_waitFences) {
        VK_CHECK_RESULT(vkCreateFence(m_logicalDevice, &fenceCreateInfo, nullptr, &fence));
    }
}

void VulkanCore::PrepareFrame()
{
    // Acquire the next image from the swap chain
    VkResult result = m_swapChain.AcquireNextImage(m_semaphores.presentComplete, &m_currentBuffer);
    // Recreate the swapchain if it's no longer compatible with the surface (OUT_OF_DATE)
    // SRS - If no longer optimal (VK_SUBOPTIMAL_KHR), wait until submitFrame() in case number of swapchain images will change on resize
    if ((result == VK_ERROR_OUT_OF_DATE_KHR) || (result == VK_SUBOPTIMAL_KHR)) {
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            WindowResize();
        }
        return;
    }
    else {
        VK_CHECK_RESULT(result);
    }
}

void VulkanCore::SubmitFrame()
{
    VkResult result = m_swapChain.QueuePresent(m_graphicsQueue, m_currentBuffer, m_semaphores.renderComplete);
    // Recreate the swapchain if it's no longer compatible with the surface (OUT_OF_DATE) or no longer optimal for presentation (SUBOPTIMAL)
    if ((result == VK_ERROR_OUT_OF_DATE_KHR) || (result == VK_SUBOPTIMAL_KHR)) {
        WindowResize();
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            return;
        }
    }
    else {
        VK_CHECK_RESULT(result);
    }

    // TODO: remove if the applications can use fences for a smaller waiting time
    VK_CHECK_RESULT(vkQueueWaitIdle(m_graphicsQueue));
}

void VulkanCore::BuildCommandBuffers()
{}

void VulkanCore::UpdateUI()
{
    ImGuiIO& io = ImGui::GetIO();

    io.DisplaySize = ImVec2((float)m_width, (float)m_height);
    io.DeltaTime = m_frameTimer;

    io.MousePos = ImVec2((float)m_mousePosX, (float)m_mousePosY);
    io.MouseDown[0] = m_mouseButtons.left && m_ui.visible;
    io.MouseDown[1] = m_mouseButtons.right && m_ui.visible;
    io.MouseDown[2] = m_mouseButtons.middle && m_ui.visible;

    ImGui::NewFrame();

    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
    ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove);

    ImGui::TextUnformatted(m_deviceProperties.deviceName);
    ImGui::Text("%.2f ms/frame (%.1d fps)", (1000.0f / m_lastFPS), m_lastFPS);

    OnUpdateUIOverlay(&m_ui);

    ImGui::End();
    ImGui::Render();

    if (m_ui.UpdateBuffers() || m_ui.updated) {
        BuildCommandBuffers();
        m_ui.updated = false;
    }
}

void VulkanCore::OnUpdateUIOverlay(VulkanIamGuiWrapper *uiWrapper)
{}

void VulkanCore::DrawUI(const VkCommandBuffer commandBuffer)
{
    if (m_ui.visible) {
        VkViewport viewport{};
        viewport.width = static_cast<float>(m_width);
        viewport.height = static_cast<float>(m_height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.extent.width = m_width;
        scissor.extent.height = m_height;
        scissor.offset.x = 0;
        scissor.offset.y = 0;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        m_ui.Draw(commandBuffer);
    }
}

void VulkanCore::WindowResize()
{
    // TODO
}
