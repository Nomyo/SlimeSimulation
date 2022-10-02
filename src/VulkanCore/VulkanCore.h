#pragma once

#include <windows.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <cstdint>
#include <chrono>
#include <vector>

#include <VulkanCamera.h>
#include <VulkanDevice.h>
#include <VulkanImguiWrapper.h>
#include <VulkanSwapChain.h>

class VulkanCore
{
public:
    VulkanCore(bool enableValidation = false);
    virtual ~VulkanCore();
    virtual void CleanUp();

    // Setup the vulkan instance, enable required extensions and connect to the physical device (GPU)
    virtual void InitVulkan();
    virtual void BuildCommandBuffers();
    virtual void CreateCommandBuffers();
    virtual void CreateSynchronizationPrimitives();
    virtual void Prepare();
    virtual void RenderLoop();
    virtual void SetupRenderPass();
    virtual void SetupWindow();
    virtual void SetupFrameBuffer();
    virtual void NextFrame();
    virtual void PrepareFrame();
    virtual void SubmitFrame();
    virtual void OnViewChanged();
    virtual void Render() = 0;

    virtual void UpdateUI();
    virtual void DrawUI(const VkCommandBuffer commandBuffer);

    // Called when the window has been resized, can be used by the sample application to recreate resources
    virtual void WindowResize();
    virtual void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    virtual void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    virtual void CursorPositionCallback(GLFWwindow* window, double xpos, double ypos);

protected:
    virtual VkResult CreateInstance(bool enableValidation);
    virtual void OnUpdateUIOverlay(VulkanIamGuiWrapper* ui);

    VkPipelineShaderStageCreateInfo LoadShader(VkDevice device, const std::string& filepath, VkShaderStageFlagBits stage);

protected:
    uint32_t m_width = 1920;
    uint32_t m_height = 1200;

    // Frame counter to display fps
    uint32_t m_frameCounter = 0;
    uint32_t m_lastFPS = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_lastTimestamp;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_tPrevEnd;

    // Last frame time measured using a high performance timer (if available)
    float  m_frameTimer = .001f;

    VkInstance m_instance;
    std::vector<std::string> m_supportedInstanceExtensions;

    // Set of device extensions to be enabled for the application, to be set in the derived object
    VkPhysicalDeviceFeatures m_enabledFeatures{};
    std::vector<const char*> m_enabledDeviceExtensions;
    std::vector<const char*> m_enabledInstanceExtensions;

    // Wrapper class around device related functionality
    VulkanDevice* m_vulkanDevice;

    // Physical device
    VkPhysicalDevice m_physicalDevice;
    VkPhysicalDeviceProperties m_deviceProperties;
    VkPhysicalDeviceFeatures m_deviceFeatures;
    VkPhysicalDeviceMemoryProperties m_deviceMemoryProperties;

    // Wraps the swap chain to present images (framebuffers) to the windowing system
    VulkanSwapChain m_swapChain;

    // Logical device
    VkDevice m_logicalDevice;

    // Handle to the device graphics queue that command buffers are submitted to
    VkQueue m_graphicsQueue;

    // Command buffer pool on swapchain graphics queue
    VkCommandPool m_cmdPool;

    // Command buffers used for rendering
    std::vector<VkCommandBuffer> m_drawCmdBuffers;

    // Global render pass for frame buffer writes
    VkRenderPass m_renderPass = VK_NULL_HANDLE;

    // List of available frame buffers (same as number of swap chain images)
    std::vector<VkFramebuffer> m_frameBuffers;
    // Active frame buffer index
    uint32_t m_currentBuffer = 0;

    std::vector<VkFence> m_waitFences;

    // Descriptor set pool
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;

    // Default clear color
    VkClearColorValue m_defaultClearColor = { { 0.0f, 0.0f, 0.0f, 1.0f } };

    /** @brief Pipeline stages used to wait at for graphics queue submissions */
    VkPipelineStageFlags m_submitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    // Contains command buffers and semaphores to be presented to the queue
    VkSubmitInfo m_submitInfo;

    // List of shader modules created (stored for cleanup)
    std::vector<VkShaderModule> m_shaderModules;

    // Ui wrapper
    VulkanIamGuiWrapper m_ui;

    // Camera
    VulkanCamera m_camera;

    // Synchronization semaphores
    struct {
        // Swap chain image presentation
        VkSemaphore presentComplete;
        // Command buffer submission and execution
        VkSemaphore renderComplete;
    } m_semaphores;

    struct {
        bool left = false;
        bool right = false;
        bool middle = false;
    } m_mouseButtons;
    double m_mousePosX;
    double m_mousePosY;

    bool m_prepared = false;
    bool m_viewUpdated = false;

    // Settings
    bool m_validation;
    std::string m_applicationName;

    GLFWwindow* m_pWindow;
};