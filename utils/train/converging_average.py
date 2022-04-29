class ExpMovingAverage:
    def __init__(self, start, alpha=0.5):
        self.avg = start
        self.alpha = alpha

    def __call__(self, *args, **kwargs):
        self.avg = self.alpha * args[0] + (1 - self.alpha) * self.avg
        return self


if __name__ == "__main__":
    moving_average = ExpMovingAverage(10)  # First loss
    moving_average(10)  # Second loss
    moving_average(10)  # Third loss so on
    print(moving_average.avg)
