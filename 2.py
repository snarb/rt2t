import main as basic_rnn
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_learning_curve(num_steps, state_size=4, epochs=1):
    global losses, total_loss, final_state, train_step, x, y, init_state
    tf.reset_default_graph()
    g = tf.get_default_graph()
    losses, total_loss, final_state, train_step, x, y, init_state = \
        basic_rnn.setup_graph(g,
            basic_rnn.RNN_config(num_steps=num_steps, state_size=state_size))
    res = basic_rnn.train_network(epochs, num_steps, state_size=state_size, verbose=False)
    plt.plot(res)
    plt.show()


if __name__ == "__main__":
    plot_learning_curve(5)

