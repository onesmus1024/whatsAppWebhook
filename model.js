const tf = require('@tensorflow/tfjs');


const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});


async function trainModel() {
    const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
    const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);
    const history=await model.fit(xs, ys, {epochs: 10,callbacks:{
        onEpochEnd: async (epoch, logs) => {
            console.log(epoch,logs.loss);
            }
    }});
    }

module.exports={
    trainModel,
    model
}