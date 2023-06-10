from pipeline import CNNPipeline

if __name__ == '__main__':
    cnn_pipeline = CNNPipeline()
    cnn_pipeline.run(epoch=10)
    cnn_pipeline.evaluate()

