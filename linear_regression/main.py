from train import train

def main():
    w, b = train(lr=0.01, num_epoch=10, batch_size=10)
    print(f"最终的权重: {w}")
    print(f"最终的偏置: {b}")

if __name__ == "__main__":
    main()
