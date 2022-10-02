#include <VulkanSwapChain.h>
#include <VulkanUtils.h>

#include <GLFW/glfw3.h>
#include <cassert>

VulkanSwapChain::VulkanSwapChain()
{
}

VulkanSwapChain::~VulkanSwapChain()
{
}

void VulkanSwapChain::CleanUp()
{
    for (auto& buffer : m_buffers) {
        vkDestroyImageView(m_logicalDevice, buffer.view, nullptr);
    }

    vkDestroySwapchainKHR(m_logicalDevice, m_swapChain, nullptr);
    vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
}

void VulkanSwapChain::Init(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice logicalDevice)
{
    m_instance = instance;
    m_logicalDevice = logicalDevice;
    m_physicalDevice = physicalDevice;
}

void VulkanSwapChain::InitSurface(GLFWwindow* pWindow)
{
    if (glfwCreateWindowSurface(m_instance, pWindow, nullptr, &m_surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }

    // Get available queue family properties
    uint32_t queueCount;
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueCount, NULL);
    assert(queueCount >= 1);

    std::vector<VkQueueFamilyProperties> queueProps(queueCount);
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueCount, queueProps.data());

    // Iterate over each queue to learn whether it supports presenting:
    // Find a queue with present support
    // Will be used to present the swap chain images to the windowing system
    std::vector<VkBool32> supportsPresent(queueCount);
    for (uint32_t i = 0; i < queueCount; i++)
    {
        vkGetPhysicalDeviceSurfaceSupportKHR(m_physicalDevice, i, m_surface, &supportsPresent[i]);
    }

    // Search for a graphics and a present queue in the array of queue
    // families, try to find one that supports both
    uint32_t graphicsQueueNodeIndex = UINT32_MAX;
    uint32_t presentQueueNodeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < queueCount; i++)
    {
        if ((queueProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0)
        {
            if (graphicsQueueNodeIndex == UINT32_MAX)
            {
                graphicsQueueNodeIndex = i;
            }

            if (supportsPresent[i] == VK_TRUE)
            {
                graphicsQueueNodeIndex = i;
                presentQueueNodeIndex = i;
                break;
            }
        }
    }
    if (presentQueueNodeIndex == UINT32_MAX)
    {
        // If there's no queue that supports both present and graphics
        // try to find a separate present queue
        for (uint32_t i = 0; i < queueCount; ++i)
        {
            if (supportsPresent[i] == VK_TRUE)
            {
                presentQueueNodeIndex = i;
                break;
            }
        }
    }

    // Exit if either a graphics or a presenting queue hasn't been found
    if (graphicsQueueNodeIndex == UINT32_MAX || presentQueueNodeIndex == UINT32_MAX)
    {
        throw("Could not find a graphics and/or presenting queue!", -1);
    }

    if (graphicsQueueNodeIndex != presentQueueNodeIndex)
    {
        throw("Separate graphics and presenting queues are not supported yet!", -1);
    }

    m_queueNodeIndex = graphicsQueueNodeIndex;

    // Get list of supported surface formats
    uint32_t formatCount;
    VK_CHECK_RESULT(vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, m_surface, &formatCount, NULL));
    assert(formatCount > 0);

    std::vector<VkSurfaceFormatKHR> surfaceFormats(formatCount);
    VK_CHECK_RESULT(vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, m_surface, &formatCount, surfaceFormats.data()));

    // If the surface format list only includes one entry with VK_FORMAT_UNDEFINED,
    // there is no preferred format, so we assume VK_FORMAT_B8G8R8A8_SRGB
    if ((formatCount == 1) && (surfaceFormats[0].format == VK_FORMAT_UNDEFINED))
    {
        m_colorFormat = VK_FORMAT_B8G8R8A8_SRGB;
        m_colorSpace = surfaceFormats[0].colorSpace;
    }
    else
    {
        // iterate over the list of available surface format and
        // check for the presence of VK_FORMAT_B8G8R8A8_SRGB
        bool found_B8G8R8A8_SRGB = false;
        for (auto&& surfaceFormat : surfaceFormats)
        {
            if (surfaceFormat.format == VK_FORMAT_B8G8R8A8_SRGB)
            {
                m_colorFormat = surfaceFormat.format;
                m_colorSpace = surfaceFormat.colorSpace;
                found_B8G8R8A8_SRGB = true;
                break;
            }
        }

        // in case VK_FORMAT_B8G8R8A8_SRGB is not available
        // select the first available color format
        if (!found_B8G8R8A8_SRGB)
        {
            m_colorFormat = surfaceFormats[0].format;
            m_colorSpace = surfaceFormats[0].colorSpace;
        }
    }
}

// Note: Width and height may need to be adjusted to fit the requirement of the swapchain thus the pointer type
void VulkanSwapChain::Create(uint32_t* width, uint32_t* height)
{
    // Store the current swap chain handle so we can use it later on to ease up recreation
    VkSwapchainKHR oldSwapchain = m_swapChain;

    // Get physical device surface properties and formats
    VkSurfaceCapabilitiesKHR surfaceCaps;
    VK_CHECK_RESULT(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physicalDevice, m_surface, &surfaceCaps));

    // Get available present modes
    uint32_t presentModeCount;
    VK_CHECK_RESULT(vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, m_surface, &presentModeCount, NULL));
    assert(presentModeCount > 0);

    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    VK_CHECK_RESULT(vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, m_surface, &presentModeCount, presentModes.data()));

    VkExtent2D swapchainExtent = {};
    // If width (and height) equals the special value 0xFFFFFFFF, the size of the surface will be set by the swapchain
    if (surfaceCaps.currentExtent.width == (uint32_t)-1)
    {
        // If the surface size is undefined, the size is set to
        // the size of the images requested.
        swapchainExtent.width = *width;
        swapchainExtent.height = *height;
    }
    else
    {
        // If the surface size is defined, the swap chain size must match
        swapchainExtent = surfaceCaps.currentExtent;
        *width = surfaceCaps.currentExtent.width;
        *height = surfaceCaps.currentExtent.height;
    }


    // Select a present mode for the swapchain

    // The VK_PRESENT_MODE_FIFO_KHR mode must always be present as per spec, non-tearing but not lowest latency
    VkPresentModeKHR swapchainPresentMode = VK_PRESENT_MODE_FIFO_KHR;

    // Try to find a mailbox mode
    // It's the lowest latency non-tearing present mode available
    for (size_t i = 0; i < presentModeCount; i++)
    {
        if (presentModes[i] == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            swapchainPresentMode = VK_PRESENT_MODE_MAILBOX_KHR;
            break;
        }
        if (presentModes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR)
        {
            swapchainPresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
        }
    }

    // Determine the number of images
    uint32_t desiredNumberOfSwapchainImages = surfaceCaps.minImageCount + 1;
    if ((surfaceCaps.maxImageCount > 0) && (desiredNumberOfSwapchainImages > surfaceCaps.maxImageCount))
    {
        desiredNumberOfSwapchainImages = surfaceCaps.maxImageCount;
    }

    // Find the transformation of the surface
    VkSurfaceTransformFlagsKHR preTransform;
    if (surfaceCaps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
    {
        // We prefer a non-rotated transform
        preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    }
    else
    {
        preTransform = surfaceCaps.currentTransform;
    }

    // Find a supported composite alpha format (not all devices support alpha opaque)
    VkCompositeAlphaFlagBitsKHR compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    // Simply select the first composite alpha format available
    std::vector<VkCompositeAlphaFlagBitsKHR> compositeAlphaFlags = {
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
    };
    for (auto& compositeAlphaFlag : compositeAlphaFlags) {
        if (surfaceCaps.supportedCompositeAlpha & compositeAlphaFlag) {
            compositeAlpha = compositeAlphaFlag;
            break;
        };
    }

    VkSwapchainCreateInfoKHR swapchainCreateInfo = {};
    swapchainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchainCreateInfo.surface = m_surface;
    swapchainCreateInfo.minImageCount = desiredNumberOfSwapchainImages;
    swapchainCreateInfo.imageFormat = m_colorFormat;
    swapchainCreateInfo.imageColorSpace = m_colorSpace;
    swapchainCreateInfo.imageExtent = { swapchainExtent.width, swapchainExtent.height };
    swapchainCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    swapchainCreateInfo.preTransform = (VkSurfaceTransformFlagBitsKHR)preTransform;
    swapchainCreateInfo.imageArrayLayers = 1;
    swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchainCreateInfo.queueFamilyIndexCount = 0;
    swapchainCreateInfo.presentMode = swapchainPresentMode;
    // Setting oldSwapChain to the saved handle of the previous swapchain aids in resource reuse and makes sure that we can still present already acquired images
    swapchainCreateInfo.oldSwapchain = oldSwapchain;
    // Setting clipped to VK_TRUE allows the implementation to discard rendering outside of the surface area
    swapchainCreateInfo.clipped = VK_TRUE;
    swapchainCreateInfo.compositeAlpha = compositeAlpha;

    // Enable transfer source on swap chain images if supported
    if (surfaceCaps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) {
        swapchainCreateInfo.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    }

    // Enable transfer destination on swap chain images if supported
    if (surfaceCaps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) {
        swapchainCreateInfo.imageUsage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    }

    VK_CHECK_RESULT(vkCreateSwapchainKHR(m_logicalDevice, &swapchainCreateInfo, nullptr, &m_swapChain));


    // If an existing swap chain is re-created, destroy the old swap chain
    // This also cleans up all the presentable images
    if (oldSwapchain != VK_NULL_HANDLE)
    {
        for (uint32_t i = 0; i < m_imageCount; i++)
        {
            vkDestroyImageView(m_logicalDevice, m_buffers[i].view, nullptr);
        }
        vkDestroySwapchainKHR(m_logicalDevice, oldSwapchain, nullptr);
    }
    VK_CHECK_RESULT(vkGetSwapchainImagesKHR(m_logicalDevice, m_swapChain, &m_imageCount, NULL));

    // Get the swap chain images
    m_images.resize(m_imageCount);
    VK_CHECK_RESULT(vkGetSwapchainImagesKHR(m_logicalDevice, m_swapChain, &m_imageCount, m_images.data()));

    // Get the swap chain buffers containing the image and imageview
    m_buffers.resize(m_imageCount);
    for (uint32_t i = 0; i < m_imageCount; i++)
    {
        VkImageViewCreateInfo colorAttachmentView = {};
        colorAttachmentView.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        colorAttachmentView.pNext = NULL;
        colorAttachmentView.format = m_colorFormat;
        colorAttachmentView.components = {
            VK_COMPONENT_SWIZZLE_R,
            VK_COMPONENT_SWIZZLE_G,
            VK_COMPONENT_SWIZZLE_B,
            VK_COMPONENT_SWIZZLE_A
        };
        colorAttachmentView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        colorAttachmentView.subresourceRange.baseMipLevel = 0;
        colorAttachmentView.subresourceRange.levelCount = 1;
        colorAttachmentView.subresourceRange.baseArrayLayer = 0;
        colorAttachmentView.subresourceRange.layerCount = 1;
        colorAttachmentView.viewType = VK_IMAGE_VIEW_TYPE_2D;
        colorAttachmentView.flags = 0;

        m_buffers[i].image = m_images[i];

        colorAttachmentView.image = m_buffers[i].image;

        VK_CHECK_RESULT(vkCreateImageView(m_logicalDevice, &colorAttachmentView, nullptr, &m_buffers[i].view));
    }
}

VkResult VulkanSwapChain::AcquireNextImage(VkSemaphore presentCompleteSemaphore, uint32_t *imageIndex)
{
    // By setting timeout to UINT64_MAX we will always wait until the next image has been acquired or an actual error is thrown
    // With that we don't have to handle VK_NOT_READY
    return vkAcquireNextImageKHR(m_logicalDevice, m_swapChain, UINT64_MAX, presentCompleteSemaphore, (VkFence)nullptr, imageIndex);
}
VkResult VulkanSwapChain::QueuePresent(VkQueue queue, uint32_t imageIndex, VkSemaphore waitSemaphore)
{
    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pNext = NULL;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &m_swapChain;
    presentInfo.pImageIndices = &imageIndex;
    // Check if a wait semaphore has been specified to wait for before presenting the image
    if (waitSemaphore != VK_NULL_HANDLE)
    {
        presentInfo.pWaitSemaphores = &waitSemaphore;
        presentInfo.waitSemaphoreCount = 1;
    }
    return vkQueuePresentKHR(queue, &presentInfo);
}

uint32_t VulkanSwapChain::GetQueueIndex()
{
    return m_queueNodeIndex;
}

uint32_t VulkanSwapChain::GetImageCount()
{
    return m_imageCount;
}

VkFormat VulkanSwapChain::GetColorFormat()
{
    return m_colorFormat;
}

VkImageView VulkanSwapChain::GetImageView(uint32_t index)
{
    if (m_buffers.size() <= index)
    {
        throw("Invalid image view index", -1);
    }
    return m_buffers[index].view;
}
