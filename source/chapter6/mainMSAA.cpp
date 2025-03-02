#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <stb_image.h>

#include <array>
#include <filesystem>
#include <gli/gli.hpp>
#include <glm/glm.hpp>
#include <tracy/Tracy.hpp>

#include "enginecore/AsyncDataUploader.hpp"
#include "enginecore/Camera.hpp"
#include "enginecore/GLBLoader.hpp"
#include "enginecore/GLFWUtils.hpp"
#include "enginecore/ImguiManager.hpp"
#include "enginecore/Model.hpp"
#include "enginecore/RingBuffer.hpp"
#include "vulkancore/Buffer.hpp"
#include "vulkancore/CommandQueueManager.hpp"
#include "vulkancore/Context.hpp"
#include "vulkancore/Framebuffer.hpp"
#include "vulkancore/Pipeline.hpp"
#include "vulkancore/RenderPass.hpp"
#include "vulkancore/Sampler.hpp"
#include "vulkancore/Texture.hpp"

// clang-format off
#include <tracy/TracyVulkan.hpp>
// clang-format on

constexpr VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_8_BIT;

GLFWwindow* window_ = nullptr;
EngineCore::Camera camera(glm::vec3(-9.f, 2.f, 2.f));
int main(int argc, char* argv[]) {
  initWindow(&window_, &camera);

#pragma region Context initialization
  std::vector<std::string> instExtension = {
      VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
      VK_KHR_SURFACE_EXTENSION_NAME,
      VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
  };

  std::vector<std::string> deviceExtension = {
#if defined(VK_EXT_calibrated_timestamps)
    VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME,
#endif
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_EXT_MEMORY_BUDGET_EXTENSION_NAME,
  };

  std::vector<std::string> validationLayers;
#ifdef _DEBUG
  validationLayers.push_back("VK_LAYER_KHRONOS_validation");
#endif

  VulkanCore::Context::enableDefaultFeatures();
  VulkanCore::Context::enableIndirectRenderingFeature();
  VulkanCore::Context::enableSynchronization2Feature();  // needed for acquire/release
                                                         // barriers
  VulkanCore::Context::enableBufferDeviceAddressFeature();

  VulkanCore::Context context((void*)glfwGetWin32Window(window_),
                              validationLayers,  // layers
                              instExtension,     // instance extensions
                              deviceExtension,   // device extensions
                              VkQueueFlags(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT),
                              true);
#pragma endregion

#pragma region Swapchain initialization
  const VkExtent2D extents =
      context.physicalDevice().surfaceCapabilities().minImageExtent;

  const VkFormat swapChainFormat = VK_FORMAT_B8G8R8A8_UNORM;

  context.createSwapchain(swapChainFormat, VK_COLORSPACE_SRGB_NONLINEAR_KHR,
                          VK_PRESENT_MODE_MAILBOX_KHR, extents);

  static const uint32_t framesInFlight = (uint32_t)context.swapchain()->numberImages();
#pragma endregion

  // Create command pools
  auto commandMgr = context.createGraphicsCommandQueue(
      context.swapchain()->numberImages(), framesInFlight, "main command");

#pragma region Tracy initialization
#if defined(VK_EXT_calibrated_timestamps)
  TracyVkCtx tracyCtx_ = TracyVkContextCalibrated(
      context.physicalDevice().vkPhysicalDevice(), context.device(),
      context.graphicsQueue(), commandMgr.getCmdBuffer(),
      vkGetPhysicalDeviceCalibrateableTimeDomainsEXT, vkGetCalibratedTimestampsEXT);
#else
  TracyVkCtx tracyCtx_ =
      TracyVkContext(context.physicalDevice().vkPhysicalDevice(), context.device(),
                     context.graphicsQueue(), commandMgr.getCmdBuffer());
#endif
#pragma endregion

  UniformTransforms transform = {.model = glm::mat4(1.0f),
                                 .view = camera.viewMatrix(),
                                 .projection = camera.getProjectMatrix()};

  constexpr uint32_t CAMERA_SET = 0;
  constexpr uint32_t TEXTURES_SET = 1;
  constexpr uint32_t SAMPLER_SET = 2;
  constexpr uint32_t STORAGE_BUFFER_SET =
      3;  // storing vertex/index/indirect/material buffer in array
  constexpr uint32_t BINDING_0 = 0;
  constexpr uint32_t BINDING_1 = 1;
  constexpr uint32_t BINDING_2 = 2;
  constexpr uint32_t BINDING_3 = 3;

  auto emptyTexture =
      context.createTexture(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 0,
                            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                            VkExtent3D{
                                .width = static_cast<uint32_t>(1),
                                .height = static_cast<uint32_t>(1),
                                .depth = static_cast<uint32_t>(1.0),
                            },
                            1, 1, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false,
                            VK_SAMPLE_COUNT_1_BIT, "Empty Texture");

  std::vector<std::shared_ptr<VulkanCore::Buffer>> buffers;
  std::vector<std::shared_ptr<VulkanCore::Texture>> textures;
  std::vector<std::shared_ptr<VulkanCore::Sampler>> samplers;
  EngineCore::RingBuffer cameraBuffer(context.swapchain()->numberImages(), context,
                                      sizeof(UniformTransforms));
  uint32_t numMeshes = 0;
  std::shared_ptr<EngineCore::Model> bistro;
  BS::thread_pool pool(std::thread::hardware_concurrency() - 2);
  pool.pause();

  std::shared_ptr<VulkanCore::Pipeline> pipeline;

  auto textureReadyCB = [&pipeline, &textures](int textureIndex, int modelId) {
    pipeline->bindResource(TEXTURES_SET, BINDING_0, 0,
                           {textures.begin() + textureIndex, 1}, nullptr, textureIndex);
  };

  EngineCore::AsyncDataUploader dataUploader(context, textureReadyCB);

  auto glbTextureDataLoadedCB = [&context, &bistro, &textures, &dataUploader](
                                    int textureIndex, int modelId) {
    EngineCore::AsyncDataUploader::TextureLoadTask t;
    textures[textureIndex] = context.createTexture(
        VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 0,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VkExtent3D{
            .width = static_cast<uint32_t>(bistro->textures[textureIndex]->width),
            .height = static_cast<uint32_t>(bistro->textures[textureIndex]->height),
            .depth = 1u,
        },
        1, 1, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true, VK_SAMPLE_COUNT_1_BIT,
        std::to_string(textureIndex));
    t.texture = textures[textureIndex].get();
    t.data = bistro->textures[textureIndex]->data;
    t.index = textureIndex;
    t.modelIndex = modelId;
    dataUploader.queueTextureUploadTasks(t);
  };

#pragma region Load model
  {
    const auto commandBuffer = commandMgr.getCmdBufferToBegin();
    {
      emptyTexture->transitionImageLayout(commandBuffer,
                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
      samplers.emplace_back(context.createSampler(
          VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT,
          VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_ADDRESS_MODE_REPEAT, 10.0f,
          "default sampler"));

      ZoneScopedN("Model load");
      EngineCore::GLBLoader glbLoader;
      bistro =
          glbLoader.load("resources/assets/Bistro.glb", pool, glbTextureDataLoadedCB);
      textures.resize(bistro->textures.size(), emptyTexture);
      TracyVkZone(tracyCtx_, commandBuffer, "Model upload");
      EngineCore::convertModel2OneBuffer(context, commandMgr, commandBuffer,
                                         *bistro.get(), buffers, samplers);
      numMeshes = bistro->meshes.size();
    }

    TracyVkCollect(tracyCtx_, commandBuffer);
    commandMgr.endCmdBuffer(commandBuffer);

    VkPipelineStageFlags flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
    const auto submitInfo =
        context.swapchain()->createSubmitInfo(&commandBuffer, &flags, false, false);
    commandMgr.submit(&submitInfo);
    commandMgr.waitUntilSubmitIsComplete();
  }
#pragma endregion

#pragma region Multisample Intermediary Texture
  // Create multisample image
  std::vector<std::shared_ptr<VulkanCore::Texture>> multisampleTextures(
      context.swapchain()->numberImages());
  for (uint32_t i = 0; i < context.swapchain()->numberImages(); ++i) {
    multisampleTextures[i] = context.createTexture(
        VK_IMAGE_TYPE_2D, swapChainFormat, 0, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        VkExtent3D(extents.width, extents.height, 1), 1, 1,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false, msaaSamples,
        "main render multisample image");
  }
#pragma endregion

#pragma region DepthTexture
  auto depthTexture = context.createTexture(
      VK_IMAGE_TYPE_2D, VK_FORMAT_D24_UNORM_S8_UINT, 0,
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
      {
          .width = context.swapchain()->extent().width,
          .height = context.swapchain()->extent().height,
          .depth = 1,
      },
      1, 1, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false, msaaSamples, "depth buffer");
#pragma endregion

  const auto resourcesFolder = std::filesystem::current_path() / "resources/shaders/";

  auto vertexShader =
      context.createShaderModule((resourcesFolder / "indirectdraw.vert").string(),
                                 VK_SHADER_STAGE_VERTEX_BIT, "main vertex");
  auto fragmentShader =
      context.createShaderModule((resourcesFolder / "indirectdraw.frag").string(),
                                 VK_SHADER_STAGE_FRAGMENT_BIT, "main fragment");

  const std::vector<VulkanCore::Pipeline::SetDescriptor> setLayout = {
      {
          .set_ = CAMERA_SET,  // set number
          .bindings_ =
              {
                  // vector of bindings
                  VkDescriptorSetLayoutBinding(
                      0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT),
              },
      },
      {
          .set_ = TEXTURES_SET,  // set number
          .bindings_ =
              {
                  // vector of bindings
                  VkDescriptorSetLayoutBinding(
                      0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000,
                      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT),
              },
      },
      {
          .set_ = SAMPLER_SET,  // set number
          .bindings_ =
              {
                  // vector of bindings
                  VkDescriptorSetLayoutBinding(
                      0, VK_DESCRIPTOR_TYPE_SAMPLER, 1000,
                      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT),
              },
      },
      {
          .set_ = STORAGE_BUFFER_SET,
          .bindings_ =
              {
                  VkDescriptorSetLayoutBinding(
                      0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4,
                      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT),
              },
      },
  };
  const VulkanCore::Pipeline::GraphicsPipelineDescriptor gpDesc = {
      .sets_ = setLayout,
      .vertexShader_ = vertexShader,
      .fragmentShader_ = fragmentShader,
      .dynamicStates_ = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                         VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE},
      .colorTextureFormats = {swapChainFormat},
      .depthTextureFormat = depthTexture->vkFormat(),
      .sampleCount = msaaSamples,
      .cullMode = VK_CULL_MODE_NONE,
      .viewport = context.swapchain()->extent(),
      .depthTestEnable = true,
      .depthWriteEnable = true,
      .depthCompareOperation = VK_COMPARE_OP_LESS,
  };

  std::vector<std::unique_ptr<VulkanCore::Framebuffer>> swapchain_framebuffers(
      context.swapchain()->numberImages());

#pragma region Render Pass Initialization
  std::shared_ptr<VulkanCore::RenderPass> renderPass = context.createRenderPass(
      {multisampleTextures[0], depthTexture},
      {VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_LOAD_OP_CLEAR,
       VK_ATTACHMENT_LOAD_OP_CLEAR},
      {VK_ATTACHMENT_STORE_OP_STORE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
       VK_ATTACHMENT_STORE_OP_STORE},
      {VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
       VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR},
      VK_PIPELINE_BIND_POINT_GRAPHICS, {context.swapchain()->texture(0)},
      "main multisample pass ");
#pragma endregion

#pragma region Swapchain Framebuffers Initialization
  for (size_t index = 0; index < context.swapchain()->numberImages(); ++index) {
    swapchain_framebuffers[index] = context.createFramebuffer(
        renderPass->vkRenderPass(),
        {multisampleTextures[index], depthTexture, context.swapchain()->texture(index)},
        nullptr, nullptr,
        "swapchain framebuffer " + std::to_string(swapchain_framebuffers.size()));
  }
#pragma endregion

#pragma region Pipeline initialization
  pipeline = context.createGraphicsPipeline(gpDesc, renderPass->vkRenderPass(), "main");
  pipeline->allocateDescriptors({
      {.set_ = CAMERA_SET, .count_ = 3},
      {.set_ = TEXTURES_SET, .count_ = 1},
      {.set_ = SAMPLER_SET, .count_ = 1},
      {.set_ = STORAGE_BUFFER_SET, .count_ = 1},
  });
  pipeline->bindResource(CAMERA_SET, BINDING_0, 0, cameraBuffer.buffer(0), 0,
                         sizeof(UniformTransforms), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
  pipeline->bindResource(CAMERA_SET, BINDING_0, 1, cameraBuffer.buffer(1), 0,
                         sizeof(UniformTransforms), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
  pipeline->bindResource(CAMERA_SET, BINDING_0, 2, cameraBuffer.buffer(2), 0,
                         sizeof(UniformTransforms), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
  pipeline->bindResource(STORAGE_BUFFER_SET, BINDING_0, 0,
                         {buffers[0], buffers[1], buffers[3],
                          buffers[2]},  // vertex, index, indirect, material
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
  pipeline->bindResource(TEXTURES_SET, BINDING_0, 0, {textures.begin(), textures.end()});
  pipeline->bindResource(SAMPLER_SET, BINDING_0, 0, {samplers.begin(), 1});
#pragma endregion

  float r = 0.6f, g = 0.6f, b = 1.f;
  size_t frame = 0;
  size_t previousFrame = 0;
  const std::array<VkClearValue, 3> clearValues = {
      VkClearValue{.color = {r, g, b, 0.0f}},
      VkClearValue{.depthStencil = {1.0f}},
      VkClearValue{.color = {r, g, b, 0.0f}},
  };

  const glm::mat4 view = glm::translate(glm::mat4(1.f), {0.f, 0.f, 0.5f});
  auto time = glfwGetTime();

  std::unique_ptr<EngineCore::GUI::ImguiManager> imguiMgr = nullptr;

  TracyPlotConfig("Swapchain image index", tracy::PlotFormatType::Number, true, false,
                  tracy::Color::Aqua);

  dataUploader.startLoadingTexturesToGPU();
  pool.unpause();

  while (!glfwWindowShouldClose(window_)) {
    const auto now = glfwGetTime();
    const auto delta = now - time;
    if (delta > 1) {
      const auto fps = static_cast<double>(frame - previousFrame) / delta;
      std::cerr << "FPS: " << fps << std::endl;
      previousFrame = frame;
      time = now;
    }

    dataUploader.processLoadedTextures(commandMgr);

    const auto texture = context.swapchain()->acquireImage();
    const auto index = context.swapchain()->currentImageIndex();
    TracyPlot("Swapchain image index", (int64_t)index);

    auto commandBuffer = commandMgr.getCmdBufferToBegin();

    const VkRenderPassBeginInfo renderpassInfo = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = renderPass->vkRenderPass(),
        .framebuffer = swapchain_framebuffers[index]->vkFramebuffer(),
        .renderArea = {.offset =
                           {
                               0,
                               0,
                           },
                       .extent =
                           {
                               .width = texture->vkExtents().width,
                               .height = texture->vkExtents().height,
                           }},
        .clearValueCount = static_cast<uint32_t>(clearValues.size()),
        .pClearValues = clearValues.data(),
    };

    if (!imguiMgr) {
      imguiMgr = std::make_unique<EngineCore::GUI::ImguiManager>(
          window_, context, commandBuffer,
          renderPass ? renderPass->vkRenderPass() : VK_NULL_HANDLE, msaaSamples);
    }

    vkCmdBeginRenderPass(commandBuffer, &renderpassInfo, VK_SUBPASS_CONTENTS_INLINE);

    if (imguiMgr) {
      imguiMgr->frameBegin();
      imguiMgr->createMenu();
      imguiMgr->createDummyText();
      imguiMgr->frameEnd();
    }

#pragma region Dynamic States
    const VkViewport viewport = {
        .x = 0.0f,
        .y = static_cast<float>(context.swapchain()->extent().height),
        .width = static_cast<float>(context.swapchain()->extent().width),
        .height = -static_cast<float>(context.swapchain()->extent().height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    const VkRect2D scissor = {
        .offset =
            {
                0,
                0,
            },
        .extent = context.swapchain()->extent(),
    };
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
    vkCmdSetDepthTestEnable(commandBuffer, VK_TRUE);
#pragma endregion

#pragma region Render
    pipeline->bind(commandBuffer);

    if (camera.isDirty()) {
      transform.view = camera.viewMatrix();
      camera.setNotDirty();
    }
    cameraBuffer.buffer()->copyDataToBuffer(&transform, sizeof(UniformTransforms));

    pipeline->bindDescriptorSets(commandBuffer,
                                 {
                                     {.set = CAMERA_SET, .bindIdx = (uint32_t)index},
                                     {.set = TEXTURES_SET, .bindIdx = 0},
                                     {.set = SAMPLER_SET, .bindIdx = 0},
                                     {.set = STORAGE_BUFFER_SET, .bindIdx = 0},
                                 });
    pipeline->updateDescriptorSets();

    vkCmdBindIndexBuffer(commandBuffer, buffers[1]->vkBuffer(), 0, VK_INDEX_TYPE_UINT32);

    vkCmdDrawIndexedIndirect(commandBuffer, buffers[3]->vkBuffer(), 0, numMeshes,
                             sizeof(EngineCore::IndirectDrawCommandAndMeshData));

#pragma endregion

    if (imguiMgr) {
      imguiMgr->recordCommands(commandBuffer);
    }

    vkCmdEndRenderPass(commandBuffer);

    TracyVkCollect(tracyCtx_, commandBuffer);

    commandMgr.endCmdBuffer(commandBuffer);

    VkPipelineStageFlags flags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    const auto submitInfo = context.swapchain()->createSubmitInfo(&commandBuffer, &flags);
    commandMgr.submit(&submitInfo);
    commandMgr.goToNextCmdBuffer();

    context.swapchain()->present();
    glfwPollEvents();

    ++frame;

    cameraBuffer.moveToNextBuffer();

    FrameMarkNamed("main frame");
  }

  vkDeviceWaitIdle(context.device());
  imguiMgr.reset();

  return 0;
}

// #define GLFW_EXPOSE_NATIVE_WIN32
// #define GLFW_EXPOSE_NATIVE_WGL
// #include <GLFW/glfw3.h>
// #define GLFW_EXPOSE_NATIVE_WIN32
// #include <GLFW/glfw3native.h>
// #include <stb_image.h>
//
// #include <array>
// #include <filesystem>
// #include <gli/gli.hpp>
// #include <glm/glm.hpp>
//
// #include "enginecore/FPSCounter.hpp"
// #include "vulkancore/CommandQueueManager.hpp"
// #include "vulkancore/Context.hpp"
// #include "vulkancore/Framebuffer.hpp"
// #include "vulkancore/RenderPass.hpp"
// #include "vulkancore/Texture.hpp"
//
// GLFWwindow* window_ = nullptr;
//
// bool initWindow(GLFWwindow** outWindow) {
//   if (!glfwInit()) return false;
//
//   glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
//   const char* title = "Chapter 1 - Triangle";
//
//   uint32_t posX = 200;
//   uint32_t posY = 200;
//   uint32_t width = 800;
//   uint32_t height = 600;
//
//   glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
//
//   GLFWwindow* window = glfwCreateWindow(width, height, title, nullptr, nullptr);
//
//   if (!window) {
//     glfwTerminate();
//     return false;
//   }
//
//   glfwSetWindowPos(window, posX, posY);
//
//   glfwSetErrorCallback([](int error, const char* description) {
//     printf("GLFW Error (%i): %s\n", error, description);
//   });
//
//   glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int, int action, int mods)
//   {
//     const bool pressed = action != GLFW_RELEASE;
//     if (key == GLFW_KEY_ESCAPE && pressed) {
//       glfwSetWindowShouldClose(window, GLFW_TRUE);
//     }
//     if (key == GLFW_KEY_ESCAPE && pressed) glfwSetWindowShouldClose(window, GLFW_TRUE);
//   });
//
//   if (outWindow) {
//     *outWindow = window;
//   }
//
//   return true;
// }
//
// int main(int argc, char** argv) {
//   initWindow(&window_);
//
//   std::vector<std::string> validationLayers;
// #ifdef _DEBUG
//   validationLayers.push_back("VK_LAYER_KHRONOS_validation");
// #endif
//
//   // Create Context
//   VulkanCore::Context::enableDefaultFeatures();
//   VulkanCore::Context context((void*)glfwGetWin32Window(window_),
//                               validationLayers,  // layers
//                               {
//                                   VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
//                                   VK_KHR_SURFACE_EXTENSION_NAME,
//                                   VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
//                                   VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
//                               },                                  // instance
//                               extensions {VK_KHR_SWAPCHAIN_EXTENSION_NAME},  // device
//                               extensions VK_QUEUE_GRAPHICS_BIT,  // request a graphics
//                               queue only true);
//
//   // Create Swapchain
//   const VkExtent2D extents =
//       context.physicalDevice().surfaceCapabilities().minImageExtent;
//
//   const VkFormat swapChainFormat = VK_FORMAT_B8G8R8A8_UNORM;
//   context.createSwapchain(swapChainFormat, VK_COLORSPACE_SRGB_NONLINEAR_KHR,
//                           VK_PRESENT_MODE_FIFO_KHR, extents);
//   const VkRect2D renderArea = {.offset = {.x = 0, .y = 0}, .extent = extents};
//
//   // Create multisample image
//   auto multisampleTexture = context.createTexture(
//       VK_IMAGE_TYPE_2D, swapChainFormat, 0, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
//       VkExtent3D(extents.width, extents.height, 1), 1, 1,
//       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false, VK_SAMPLE_COUNT_8_BIT,
//       "main render multisample image");
//
//   // Create Shader Modules
//   const auto shadersPath = std::filesystem::current_path() / "resources/shaders";
//   const auto vertexShaderPath = shadersPath / "triangle.vert";
//   const auto fragShaderPath = shadersPath / "triangle.frag";
//   const auto vertexShader =
//       context.createShaderModule(vertexShaderPath.string(),
//       VK_SHADER_STAGE_VERTEX_BIT);
//   const auto fragShader =
//       context.createShaderModule(fragShaderPath.string(),
//       VK_SHADER_STAGE_FRAGMENT_BIT);
//
//   // Create Framebuffers
//   std::vector<std::shared_ptr<VulkanCore::Framebuffer>> swapchain_framebuffers(
//       context.swapchain()->numberImages());
//
//   // Create Render Pass
//   std::shared_ptr<VulkanCore::RenderPass> renderPass = context.createRenderPass(
//       {multisampleTexture}, {VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_LOAD_OP_CLEAR},
//       {VK_ATTACHMENT_STORE_OP_STORE, VK_ATTACHMENT_STORE_OP_STORE},
//       {VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR},
//       VK_PIPELINE_BIND_POINT_GRAPHICS, "main multisample",
//       {context.swapchain()->texture(0)});
//
//   const VkViewport viewport = {
//       .x = 0.0f,
//       .y = static_cast<float>(context.swapchain()->extent().height),
//       .width = static_cast<float>(context.swapchain()->extent().width),
//       .height = -static_cast<float>(context.swapchain()->extent().height),
//       .minDepth = 0.0f,
//       .maxDepth = 1.0f,
//   };
//
//   // Create Graphics Pipeline
//   auto pipeline = context.createGraphicsPipeline(
//       VulkanCore::Pipeline::GraphicsPipelineDescriptor{
//           .vertexShader_ = vertexShader,
//           .fragmentShader_ = fragShader,
//           .colorTextureFormats = {swapChainFormat},
//           .sampleCount = VK_SAMPLE_COUNT_8_BIT,
//           .viewport = viewport,
//           .depthTestEnable = false,
//       },
//       renderPass->vkRenderPass());
//
//   // Create Command Manager
//   auto commandMgr = context.createGraphicsCommandQueue(
//       context.swapchain()->numberImages(), context.swapchain()->numberImages());
//
//   // FPS Counter
//   EngineCore::FPSCounter fps(glfwGetTime());
//
//   // Main Render Loop
//   while (!glfwWindowShouldClose(window_)) {
//     fps.update(glfwGetTime());
//
//     const auto texture = context.swapchain()->acquireImage();
//     const auto swapchainImageIndex = context.swapchain()->currentImageIndex();
//
//     // Create the framebuffer the first time we get here, once for each
//     // swapchain image
//     if (swapchain_framebuffers[swapchainImageIndex] == nullptr) {
//       swapchain_framebuffers[swapchainImageIndex] = context.createFramebuffer(
//           renderPass->vkRenderPass(), {multisampleTexture, texture}, nullptr, nullptr);
//     }
//
//     auto commandBuffer = commandMgr.getCmdBufferToBegin();
//
//     // Begin Render Pass
//     constexpr std::array<VkClearValue, 2> clearColors = {
//         VkClearValue{0.0f, 0.0f, 0.0f, 0.0f},
//         VkClearValue{0.0f, 0.0f, 0.0f, 0.0f},
//     };
//     const VkRenderPassBeginInfo renderpassInfo = {
//         .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
//         .renderPass = renderPass->vkRenderPass(),
//         .framebuffer = swapchain_framebuffers[swapchainImageIndex]->vkFramebuffer(),
//         .renderArea = renderArea,
//         .clearValueCount = static_cast<uint32_t>(clearColors.size()),
//         .pClearValues = clearColors.data(),
//     };
//     vkCmdBeginRenderPass(commandBuffer, &renderpassInfo, VK_SUBPASS_CONTENTS_INLINE);
//
//     pipeline->bind(commandBuffer);
//
//     vkCmdDraw(commandBuffer, 3, 1, 0, 0);
//
//     vkCmdEndRenderPass(commandBuffer);
//
//     commandMgr.endCmdBuffer(commandBuffer);
//     constexpr VkPipelineStageFlags flags =
//     VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; const auto submitInfo =
//     context.swapchain()->createSubmitInfo(&commandBuffer, &flags);
//     commandMgr.submit(&submitInfo);
//     commandMgr.goToNextCmdBuffer();
//
//     // Present render output to the screen
//     context.swapchain()->present();
//
//     glfwPollEvents();
//
//     // Increment frame number
//     fps.incFrame();
//   }
//
//   commandMgr.waitUntilAllSubmitsAreComplete();
//
//   glfwDestroyWindow(window_);
//   glfwTerminate();
//
//   return 0;
// }
