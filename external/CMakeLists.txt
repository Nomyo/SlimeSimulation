find_package(Vulkan REQUIRED)

#### GLM
add_library(glm INTERFACE)
target_include_directories(glm INTERFACE glm)

### GLFW
add_library(glfw INTERFACE)
set(glfw_DIR $ENV{GLFW_PATH})
if (WIN32)
target_include_directories(glfw INTERFACE ${glfw_DIR}/include)
find_library(GLFW_LIB glfw3 HINTS ${glfw_DIR}/lib-vc2017)
elseif (UNIX)
endif()

target_link_libraries(glfw INTERFACE ${GLFW_LIB})

#### IMGUI
add_library(imgui STATIC)
target_include_directories(imgui PUBLIC imgui)
target_sources(imgui PRIVATE 
    imgui/imgui.h
    imgui/imgui.cpp

    imgui/imgui_demo.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_widgets.cpp

    imgui/imgui_impl_glfw.cpp
    )

target_link_libraries(imgui PUBLIC Vulkan::Vulkan glfw)


#### KTX
add_library(ktx STATIC)
target_include_directories(ktx PUBLIC ktx/include ktx/other_include)
target_sources(ktx PRIVATE 
    ktx/lib/checkheader.c
    ktx/lib/filestream.c
    ktx/lib/hashlist.c
    ktx/lib/swap.c
    ktx/lib/texture.c
    ktx/lib/memstream.c
    )
target_link_libraries(ktx PUBLIC ${KTX_LIB})
