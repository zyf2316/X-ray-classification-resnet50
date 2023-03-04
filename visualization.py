import matplotlib.pyplot as plt
import numpy as np
import os

experiment_name = "experiment_contrast"

input_path = "/home/yufei.zhang/Documents/HC/assignment_2/task_2/results"
save_fig_path = "/home/yufei.zhang/Documents/HC/assignment_2/task_2/visualize"
train_loss_path = os.path.join(input_path,experiment_name,"loss_train.txt")
train_acc_path = os.path.join(input_path,experiment_name,"acc_train.txt")

with open(train_loss_path, 'r') as file:
    train_loss = np.loadtxt(file)

with open(train_acc_path, 'r') as file:
    train_acc = np.loadtxt(file)

epoch = np.arange(0, 20)
plt.plot(epoch, train_loss)
plt.title(experiment_name + ": training loss")
plt.xlabel('epoch')
plt.ylabel('training loss')

# Save the plot as a PNG file
plt.savefig(os.path.join(save_fig_path, experiment_name,"train_loss.png"))

plt.clf()

epoch = np.arange(0, 20)
plt.plot(epoch, train_acc)
plt.title(experiment_name + ": training accuracy")
plt.xlabel('epoch')
plt.ylabel('training accuracy (%)')
# Set the x-tick labels to integer values
plt.xticks(epoch)

# Save the plot as a PNG file
plt.savefig(os.path.join(save_fig_path, experiment_name,"train_acc.png"))
