from unet_model import UNetSegmentation

if __name__ == '__main__':

    model = UNetSegmentation.get_model()
    print(model.summary())


