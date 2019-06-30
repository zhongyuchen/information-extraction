from matplotlib import pyplot as plt


def extract(log_path):
    loss = []
    acc = []
    with open(log_path, 'r') as f:
        for line in f:
            loss_split = line.split('loss: ')
            if len(loss_split) >= 2:
                time_split = loss_split[1].split(' time')
                if len(time_split) >= 2:
                    loss.append(float(time_split[0]))
            acc_split = line.split('f1=tensor(')
            if len(acc_split) >= 2:
                acc.append(float(acc_split[1].split(',')[0]))

    # return loss[2:min(len(loss), 40000):slice]
    return loss[::1], acc[::1]


def draw():
    cnn_loss, cnn_acc = extract('log/rcnn_log0.txt')

    plt.subplot(121)
    plt.plot(cnn_loss, label='patience:20')
    plt.xlabel('100 steps')
    plt.ylabel('loss')
    plt.title('loss')
    plt.legend()

    plt.subplot(122)
    plt.plot(cnn_acc[:-2], label='best dev f1: %.2f' % (max(cnn_acc) * 100))
    plt.xlabel('epoch')
    plt.ylabel('f1')
    plt.title('f1')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    draw()
