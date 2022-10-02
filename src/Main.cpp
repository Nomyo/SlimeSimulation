#include <SlimeSimulation.h>

int main() {

    SlimeSimulation *simulation = new SlimeSimulation();
    simulation->SetupWindow();
    simulation->InitVulkan();
    simulation->Prepare();
    simulation->RenderLoop();
    delete(simulation);
    return 0;
}