import numpy as np
import tensorflow as tf

# # Model parameters
# W = tf.Variable([.3], dtype=tf.float32)
# b = tf.Variable([-.3], dtype=tf.float32)
#
# # Model input and output
# x = tf.placeholder(tf.float32)
# linear_model = W * x + b
# y = tf.placeholder(tf.float32)
#
# # loss
# loss = tf.reduce_sum(tf.square(linear_model - y))   # sum of squares
#
# # optimzer
# optimizer = tf.train.GradientDescentOptimizer(0.01)
# train = optimizer.minimize(loss)
#
# # training data
# x_train = [1,2,3,4]
# y_train = [0,-1,-2,-3]
#
# # training loop
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
#
# for i in range(1000):
#     sess.run(train, {x:x_train, y:y_train})
#
# # evaluate training accuracy
# curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
# print("W: %s, b: %s, loss: %s"%(curr_W, curr_b, curr_loss))


def model(features, labels, mode):
    W = tf.get_variable("W", [1], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)
    y = W*features['x'] + b

    loss = tf.reduce_sum(tf.square(y-labels))

    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)

    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
    return tf.contrib.learn.ModelFnOps(
        mode=mode, predictions=y,
        loss=loss, train_op=train)


features = [tf.contrib.layers.real_valued_column("x", dimension=1)]
#estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)
estimator = tf.contrib.learn.Estimator(model_fn=model)

x_train = np.array([1.,2.,3.,4.])
y_train = np.array([0.,-1.,-2.,-3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x_train}, y_train, batch_size=4, num_epochs=1000)
eval_input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x_eval}, y_eval, batch_size=4, num_epochs=1000)

estimator.fit(input_fn=input_fn, steps=1000)

train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)

print("train loss: %r"% train_loss)
print("eval loss: %r"% eval_loss)
