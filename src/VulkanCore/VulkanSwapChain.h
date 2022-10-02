#pragma once

#include <vulkan/vulkan.h>

#include <vector>

typedef struct SwapChainBuffers_Rec {
    VkImage image;
    VkImageView view;
} SwapChainBuffer;

struct GLFWwindow;

class VulkanSwapChain
{
public:
    VulkanSwapChain();
    ~VulkanSwapChain();
    void CleanUp();
    void Init(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice logicalDevice);
    void InitSurface(GLFWwindow* pWindow);
    void Create(uint32_t* width, uint32_t* height);
    VkResult AcquireNextImage(VkSemaphore presentCompleteSemaphore, uint32_t *imageIndex);
    VkResult QueuePresent(VkQueue queue, uint32_t imageIndex, VkSemaphore waitSemaphore);

    uint32_t GetImageCount();
    VkFormat GetColorFormat();
    uint32_t GetQueueIndex();
    VkImageView GetImageView(uint32_t index);

private:
    VkSurfaceKHR m_surface;
    VkFormat m_colorFormat;
    VkColorSpaceKHR m_colorSpace;
    VkSwapchainKHR m_swapChain = VK_NULL_HANDLE;
    uint32_t m_imageCount;
    std::vector<VkImage> m_images;
    std::vector<SwapChainBuffer> m_buffers;
    uint32_t m_queueNodeIndex = UINT32_MAX;

    VkInstance m_instance;
    VkDevice m_logicalDevice;
    VkPhysicalDevice m_physicalDevice;
};