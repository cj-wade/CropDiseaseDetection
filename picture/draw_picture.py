import matplotlib.pyplot as plt
import numpy as np

def draw(fig_loss, fig_acc):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ln1 = ax1.plot(np.arange(len(fig_loss)), fig_loss, 'r', label='loss')
    ln2 = ax2.plot(np.arange(len(fig_acc)), fig_acc, 'g', label='accuracy')
    ax1.set_xlabel('iteration * 100')
    ax1.set_ylabel('training loss')
    ax2.set_ylabel('training accuracy')

    lns = ln1 + ln2
    # labels = ["Loss", "Accuracy"]
    labels = [l.get_label() for l in lns]
    plt.legend(lns, labels)
    plt.grid(True)

    plt.figure(2)
    plt.plot(np.arange(len(fig_loss)), fig_loss, 'r', label='loss')
    plt.legend(labels=['loss'])
    plt.grid(True)

    plt.figure(3)
    plt.plot(np.arange(len(fig_acc)), fig_acc, 'g', label='accuracy')
    plt.legend(labels=["acuracy"])

    plt.grid(True)

    plt.show()
