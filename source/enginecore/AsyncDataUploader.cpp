#include "AsyncDataUploader.hpp"

#include "vulkancore/Texture.hpp"

namespace EngineCore {

AsyncDataUploader::AsyncDataUploader(VulkanCore::Context& context,
                                     std::function<void(int, int)> textureReadyCallback)
    : context_(context),
      transferCommandQueueMgr_(context.createTransferCommandQueue(
          1, 1, "secondary thread transfer command queue")),
      textureReadyCallback_(textureReadyCallback) {}

AsyncDataUploader::~AsyncDataUploader() {
  closeThreads_ = true;
  textureGPUDataUploadThread_.join();
}

void AsyncDataUploader::startLoadingTexturesToGPU() {
  // should be able to replace this with BS_Thread_pool
  textureGPUDataUploadThread_ = std::thread([this]() {
    while (!closeThreads_) {
      if (textureLoadTasks_.size() > 0) {
        textureLoadTasksMutex_.lock();
        // pop &  do stuff
        auto textureLoadTask = textureLoadTasks_.front();
        textureLoadTasks_.pop_front();
        textureLoadTasksMutex_.unlock();

        auto textureUploadStagingBuffer = context_.createStagingBuffer(
            textureLoadTask.texture->vkDeviceSize(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            "Async texture upload staging buffer");

        const auto commandBuffer = transferCommandQueueMgr_.getCmdBufferToBegin();

          textureLoadTask.texture->uploadOnly(commandBuffer,
                                              textureUploadStagingBuffer.get(),
                                              textureLoadTask.data, 0, 1);

        textureLoadTask.texture->addReleaseBarrier(
            commandBuffer, transferCommandQueueMgr_.queueFamilyIndex(),
            context_.physicalDevice().graphicsFamilyIndex().value());

        transferCommandQueueMgr_.endCmdBuffer(commandBuffer);

        transferCommandQueueMgr_.disposeWhenSubmitCompletes(
            std::move(textureUploadStagingBuffer));

        VkSemaphore graphicsSemaphore;
        const VkSemaphoreCreateInfo semaphoreInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        };
        VK_CHECK(vkCreateSemaphore(context_.device(), &semaphoreInfo, nullptr,
                                   &graphicsSemaphore));

        VkPipelineStageFlags flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
        auto submitInfo =
            context_.swapchain()->createSubmitInfo(&commandBuffer, &flags, false, false);
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &graphicsSemaphore;
        transferCommandQueueMgr_.submit(&submitInfo);
        transferCommandQueueMgr_.waitUntilSubmitIsComplete();

        mipGenMutex_.lock();
        textureMipGenerationTasks_.push_back(
            {textureLoadTask.texture, graphicsSemaphore, textureLoadTask.index});
        mipGenMutex_.unlock();
      }
    }
  });

}

void AsyncDataUploader::processLoadedTextures(
    VulkanCore::CommandQueueManager& graphicsCommandQueueMgr) {

    if (textureMipGenerationTasks_.size() > 0) {
        mipGenMutex_.lock();
        // pop &  do stuff
        auto task = textureMipGenerationTasks_.front();
        textureMipGenerationTasks_.pop_front();
        mipGenMutex_.unlock();

        auto commandBuffer = graphicsCommandQueueMgr.getCmdBufferToBegin();
        task.texture->addAcquireBarrier(commandBuffer,
                                        transferCommandQueueMgr_.queueFamilyIndex(),
                                        graphicsCommandQueueMgr.queueFamilyIndex());
        { 
            
            task.texture->generateMips(commandBuffer); 
        }


        graphicsCommandQueueMgr.endCmdBuffer(commandBuffer);
        VkPipelineStageFlags flags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        auto submitInfo =
            context_.swapchain()->createSubmitInfo(&commandBuffer, &flags, false, false);
        submitInfo.pWaitSemaphores = &task.graphicsSemaphore;
        submitInfo.waitSemaphoreCount = 1;
        graphicsCommandQueueMgr.submit(&submitInfo);

        graphicsCommandQueueMgr.waitUntilSubmitIsComplete();

        vkDestroySemaphore(context_.device(), task.graphicsSemaphore, nullptr);

        textureReadyCallback_(task.index, 0);
    }
    
}

void AsyncDataUploader::queueTextureUploadTasks(const TextureLoadTask& textureLoadTask) {
  textureLoadTasksMutex_.lock();
  textureLoadTasks_.push_back(textureLoadTask);
  textureLoadTasksMutex_.unlock();
}

}  // namespace EngineCore
