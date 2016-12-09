import numpy as np
import tensorflow as tf
import csv
import os


# Transform image to vector
def processState(state1):
    return np.reshape(state1, [32*32])


# Updates target graph with parameters from the main graph
def updateTargetGraph(tfVars):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:int(total_vars/2)]):
        op_holder.append(tfVars[idx+int(total_vars/2)].assign(var.value()))
    return op_holder


# Updates target network with parameters from main network
def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)
    total_vars = len(tf.trainable_variables())
    a = tf.trainable_variables()[0].eval(session=sess)
    b = tf.trainable_variables()[int(total_vars/2)].eval(session=sess)
    if a.all() == b.all():
        print("Target Set Success")
    else:
        print("Target Set Failed")


def convert_list_of_ints_to_string(list_of_ints):
    return (str(list_of_ints).strip('[]')).replace(" ", "")


def write_results_to_file(file_index, species, probs):
    # Make a path for our results to be saved to
    if not os.path.exists('results'):
        os.makedirs('results')
    print('Writing results to file')
    with open('results/results{}.csv'.format(file_index), 'w') as f1:
        writer = csv.writer(f1, delimiter=' ', escapechar=' ', quoting=csv.QUOTE_NONE)
        header = 'id,' + ','.join(species)
        writer.writerow([header])
        for row in probs:
            writer.writerow([row])
    print('Successfully wrote results to file')
