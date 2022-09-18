const tf = require('@tensorflow/tfjs');


const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});


async function trainModel() {
    const xs = tf.tensor2d([2, 8, 10, 12], [4, 1]);
    const ys = tf.tensor2d([12, 42, 52, 62], [4, 1]);
    const history=await model.fit(xs, ys, {epochs: 20,callbacks:{
        onEpochEnd: async (epoch, logs) => {
            console.log(epoch,logs.loss);
            }
    }});
    }

module.exports={
    trainModel,
    model
}