import matplotlib.pyplot as plt
import seaborn as sns

def plot_loss(histories, name, key='loss'):
    plt.figure(figsize=(8,4))

    for label, history in histories:
        plt.plot(history.epoch, history.history['val_'+key], color='blue', label=label.title()+' Val')
        plt.plot(history.epoch, history.history[key], color='green', label=label.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0,max(history.epoch)])
    plt.ylim([0,0.1])
    #plt.yscale('log')
    plt.savefig('plots/pose_loss_{}.png'.format(name))
    #plt.show()
    plt.close() 