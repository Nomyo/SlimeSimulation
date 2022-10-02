#pragma once

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>


class VulkanCamera
{
public:
    VulkanCamera();
    VulkanCamera(glm::vec3 pos, glm::vec3 rotation, float fov, float aspect, float znear, float zfar);
    ~VulkanCamera();

    // deltaTime from the previous frame
    void Update(float deltaTime);
    bool HasViewChanged();

    // Actions
    void SetPerspective(float fov, float aspect, float znear, float zfar);
    void SetPosition(glm::vec3 position);
    void SetRotation(glm::vec3 rotation);
    void Rotate(glm::vec3 delta);
    void SetTranslation(glm::vec3 translation);
    void Translate(glm::vec3 delta);
    void SetRotationSpeed(float rotationSpeed);
    void SetMovementSpeed(float movementSpeed);

    glm::mat4 GetViewMatrix() { return m_matrices.view; }
    glm::mat4 GetProjectionMatrix() { return m_matrices.perspective; }

    float GetRotationSpeed() { return m_rotationSpeed; }
    float GetMovementsSpeed() { return m_movementSpeed; }

    struct
    {
        bool left = false;
        bool right = false;
        bool up = false;
        bool down = false;
        bool space = false;
        bool ctrl = false;
    } keys;

private:
    void UpdateViewMatrix();
    bool Moving();

    float m_fov;
    float m_znear;
    float m_zfar;

    float m_rotationSpeed = 1.0f;
    float m_movementSpeed = 1.0f;

    bool updated = false;

    struct
    {
        glm::mat4 perspective;
        glm::mat4 view;
    } m_matrices;

    glm::vec3 m_rotation;
    glm::vec3 m_position;
};

