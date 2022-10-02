add_executable(SlimeSimulation
    Main.cpp
    SlimeSimulation.cpp
    VulkanCore/VulkanCamera.cpp
    VulkanCore/VulkanCore.cpp
    VulkanCore/VulkanImguiWrapper.cpp
    VulkanCore/VulkanSwapChain.cpp
    VulkanCore/VulkanTexture.cpp
    VulkanCore/VulkanDevice.cpp)


set_property(TARGET SlimeSimulation PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:SlimeSimulation>")

target_include_directories(SlimeSimulation PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" VulkanCore)
target_link_libraries(SlimeSimulation glm ktx imgui glfw)

target_link_libraries(SlimeSimulation Vulkan::Vulkan)

add_dependencies(SlimeSimulation Shaders)
