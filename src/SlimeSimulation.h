#pragma once

#include <VulkanCore.h>
#include <VulkanTexture.h>
#include <glm/glm.hpp>

#define ENABLE_VALIDATION true
#define AGENT_COUNT 10

struct SlimeAgent {
    glm::vec2 position;
    float angle;
};

struct BufferWrapper {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDescriptorBufferInfo descriptor;
    void *mapped = nullptr;
};

class SlimeSimulation : public VulkanCore
{
public:
    struct pipelineWrapper {
        VkDescriptorSetLayout descriptorSetLayout;
        VkDescriptorSet descriptorSet;
        VkPipeline pipeline;
        VkPipelineLayout pipelineLayout;
    };

    // Resources for the graphics part
    struct {
        uint32_t queueFamilyIndex;

        // Slime pipeline
        pipelineWrapper slime;

        // Execution dependency between compute & graphic submission
        VkSemaphore semaphore;
    } m_graphics;

    struct {
        uint32_t queueFamilyIndex;
        VkQueue queue;
        VkCommandPool commandPool;
        VkCommandBuffer commandBuffer;

        // Slime pipeline
        pipelineWrapper slime;
        // Diffuse
        pipelineWrapper diffuse;

        // Store the slime agent informations
        BufferWrapper storageBuffer;

        // Store ubo info
        BufferWrapper uniformBuffer;

        // Execution dependency between compute & graphic submission
        VkSemaphore semaphore;

        struct computeUbo {
            float elapsedTime;
            float time;
            uint32_t spaceWidth;
            uint32_t spaceHeight;
            uint32_t agentCount;
        } ubo;
    } m_compute;

    struct {
        // Texture sampled to full screen
        Texture2D renderMap;
        Texture2D diffuseMap;
    } m_textures;

    SlimeSimulation();
    virtual ~SlimeSimulation();

    virtual void Render();
    virtual void Prepare();
private:

    void PrepareGraphics();
    void PrepareCompute();

    void PrepareGraphicsPipelines();
    void PrepareGraphicsSlimePipeline();

    void PrepareComputeSlimePipeline();
    void PrepareComputeDiffusePipeline();

    void PrepareStorageBuffers();
    void PrepareUniformBuffers();

    void Draw();

    void SetupDescriptorPool();
    void SetupGraphicsDescriptorSetLayout();
    void SetupGraphicsDescriptorSet();

    virtual void BuildCommandBuffers();
    void BuildComputeCommandBuffer();
    void UpdateUniformBuffers();

    virtual void OnUpdateUIOverlay(VulkanIamGuiWrapper* ui);
    virtual void OnViewChanged();

    void CopyDiffuseToRenderTexture(VkCommandBuffer cmdBuffer);


    std::vector<VkFence> m_queueCompleteFences;
};
