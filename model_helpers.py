import numpy as np
import tensorflow as tf


# Transform image to vector
def processState(state1):
    return np.reshape(state1, [80*80])


# Updates target graph with parameters from the main graph
def updateTargetGraph(tfVars):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars/2]):
        op_holder.append(tfVars[idx+total_vars/2].assign(var.value()))
    return op_holder


# Updates target network with parameters from main network
def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)
    total_vars = len(tf.trainable_variables())
    a = tf.trainable_variables()[0].eval(session=sess)
    b = tf.trainable_variables()[total_vars/2].eval(session=sess)
    if a.all() == b.all():
        print("Target Set Success")
    else:
        print("Target Set Failed")
