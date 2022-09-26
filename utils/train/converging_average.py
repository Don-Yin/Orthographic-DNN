class ExpMovingAverage:
    def __init__(self, start, alpha=0.5):
        self.avg = start
        self.alpha = alpha

    def __call__(self, *args, **kwargs):
        self.avg = self.alpha * args[0] + (1 - self.alpha) * self.avg
        return self


if __name__ == "__main__":
    moving_average = ExpMovingAverage(10)  # First loss
    moving_average(5)  # Second loss
    moving_average(3)  # Third loss so on
    moving_average(2)
    moving_average(1)
    moving_average(0.3)
    print(moving_average.avg)
