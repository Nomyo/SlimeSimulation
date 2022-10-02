#pragma once

#include <vulkan/vulkan.h>

#include <ktx.h>
#include <ktxvulkan.h>

#include <string>

struct VulkanDevice;
class Texture
{
public:
    VulkanDevice* m_device;
    VkImage m_image;
    VkImageView m_imageView;
    VkImageLayout m_imageLayout;
    VkDeviceMemory m_deviceMemory;

    VkDescriptorImageInfo m_descriptor;
    VkSampler m_sampler;

    uint32_t m_width;
    uint32_t m_height;

    uint32_t m_mipLevels;
    uint32_t m_layerCount;

    void UpdateDescriptor();
    void Destroy();
    ktxResult loadKTXFile(std::string filename, ktxTexture **target);
};

class Texture2D : public Texture
{
public:
    // Load a 2D texture including all mip levels
    // filename must be in .ktx format
    void LoadFromFile(
        const std::string& filename,
        VkFormat           format,
        VulkanDevice      *device,
        VkQueue            copyQueue,
        VkImageUsageFlags  imageUsageFlags = VK_IMAGE_USAGE_SAMPLED_BIT,
        VkImageLayout      imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        bool               forceLinear = false);

    void CreateTargetTexture(
        uint32_t width,
        uint32_t height,
        VkFormat format,
        VulkanDevice *device,
        VkQueue copyQueue,
        VkImageUsageFlags imageUsageFlags = VK_IMAGE_USAGE_SAMPLED_BIT,
        VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
};