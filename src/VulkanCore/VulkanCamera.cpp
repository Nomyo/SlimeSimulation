#include <VulkanCamera.h>

VulkanCamera::VulkanCamera()
{
}

VulkanCamera::~VulkanCamera()
{}

VulkanCamera::VulkanCamera(glm::vec3 position, glm::vec3 rotation, float fov, float aspect, float znear, float zfar)
    : m_position(position)
    , m_rotation(rotation)
    , m_fov(fov)
    , m_znear(znear)
    , m_zfar(zfar)
{
    UpdateViewMatrix();
    SetPerspective(fov, aspect, znear, zfar);
}

void VulkanCamera::UpdateViewMatrix()
{
    glm::mat4 rotM = glm::mat4(1.0f);
    glm::mat4 transM;

    // Handle rotation per axis
    rotM = glm::rotate(rotM, glm::radians(m_rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
    rotM = glm::rotate(rotM, glm::radians(m_rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
    rotM = glm::rotate(rotM, glm::radians(m_rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

    glm::vec3 translation = m_position;
    transM = glm::translate(glm::mat4(1.0f), translation);

    m_matrices.view = rotM * transM;

    updated = true;
}

void VulkanCamera::Update(float deltaTime)
{
    updated = false;
    if (Moving())
    {
        glm::vec3 camFront;
        camFront.x = -cos(glm::radians(m_rotation.x)) * sin(glm::radians(m_rotation.y));
        camFront.y = sin(glm::radians(m_rotation.x));
        camFront.z = cos(glm::radians(m_rotation.x)) * cos(glm::radians(m_rotation.y));
        camFront = glm::normalize(camFront);

        float moveSpeed = deltaTime * m_movementSpeed;

        if (keys.up)
        {
            m_position += camFront * moveSpeed;
        }
        if (keys.down)
        {
            m_position -= camFront * moveSpeed;
        }
        if (keys.left)
        {
            m_position -= glm::normalize(glm::cross(camFront, glm::vec3(0.0f, 1.0f, 0.0f))) * moveSpeed;
        }
        if (keys.right)
        {
            m_position += glm::normalize(glm::cross(camFront, glm::vec3(0.0f, 1.0f, 0.0f))) * moveSpeed;
        }
        if (keys.ctrl)
        {
            m_position -= glm::normalize(glm::vec3(0.0f, 1.0f, 0.0f)) * moveSpeed;
        }
        if (keys.space)
        {
            m_position += glm::normalize(glm::vec3(0.0f, 1.0f, 0.0f)) * moveSpeed;
        }

        UpdateViewMatrix();
    }
}

bool VulkanCamera::HasViewChanged()
{
    return updated;
}

bool VulkanCamera::Moving()
{
    return keys.left || keys.right || keys.up || keys.down || keys.space || keys.ctrl;
}

void VulkanCamera::SetPerspective(float fov, float aspect, float znear, float zfar)
{
    // change previous perspective setup
    m_fov = fov;
    m_znear = znear;
    m_zfar = zfar;
    m_matrices.perspective = glm::perspective(glm::radians(fov), aspect, znear, zfar);
};

void VulkanCamera::SetPosition(glm::vec3 position)
{
    m_position = position;
    UpdateViewMatrix();
}

void VulkanCamera::SetRotation(glm::vec3 rotation)
{
    m_rotation = rotation;
    UpdateViewMatrix();
}

void VulkanCamera::VulkanCamera::Rotate(glm::vec3 delta)
{
    m_rotation += delta;
    UpdateViewMatrix();
}

void VulkanCamera::SetTranslation(glm::vec3 translation)
{
    m_position = translation;
    UpdateViewMatrix();
};

void VulkanCamera::Translate(glm::vec3 delta)
{
    m_position += delta;
    UpdateViewMatrix();
}

void VulkanCamera::SetRotationSpeed(float rotationSpeed)
{
    m_rotationSpeed = rotationSpeed;
}

void VulkanCamera::SetMovementSpeed(float movementSpeed)
{
    m_movementSpeed = movementSpeed;
}
