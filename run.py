from pipeline import CNNPipeline

if __name__ == '__main__':
    cnn_pipeline = CNNPipeline()
    # cnn_pipeline.run(epoch=20)
    cnn_pipeline.evaluate()

