import {TRAINING_DATA} from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js';


// Grab a reference to the MNIST input values (pixel data).

const INPUTS = TRAINING_DATA.inputs;


// Grab reference to the MNIST output values.

const OUTPUTS = TRAINING_DATA.outputs;


// Shuffle the two arrays in the same way so inputs still match outputs indexes.

tf.util.shuffleCombo(INPUTS, OUTPUTS);


// Input feature Array is 1 dimensional.

const INPUTS_TENSOR = tf.tensor2d(INPUTS);


// Output feature Array is 1 dimensional.

const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, 'int32'), 10);

// Now actually create and define model architecture.

const model = tf.sequential();


model.add(tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}));

model.add(tf.layers.dense({units: 16, activation: 'relu'}));

model.add(tf.layers.dense({units: 10, activation: 'softmax'}));


model.summary();


train();

async function train() { 

  // Compile the model with the defined optimizer and specify our loss function to use.

  model.compile({

    optimizer: 'adam',

    loss: 'categoricalCrossentropy',

    metrics: ['accuracy']

  });


  let results = await model.fit(INPUTS_TENSOR, OUTPUTS_TENSOR, {

    shuffle: true,        // Ensure data is shuffled again before using each epoch.

    validationSplit: 0.2,

    batchSize: 512,       // Update weights after every 512 examples.experiment with different batch size  

    epochs: 50,           // Go over the data 50 times! experiment with different epoch size

    callbacks: {onEpochEnd: logProgress}

  });

  

  OUTPUTS_TENSOR.dispose();

  INPUTS_TENSOR.dispose();

  evaluate(); // Once trained we can evaluate the model.

}


const PREDICTION_ELEMENT = document.getElementById('prediction');


function evaluate() {

  const OFFSET = Math.floor((Math.random() * INPUTS.length)); // Select random from all example inputs. 

 

  let answer = tf.tidy(function() {

    let newInput = tf.tensor1d(INPUTS[OFFSET]).expandDims(); // the ‘model.predict’ function expects a batch of images as input. To avoid an error, 
    //call expandDims on the tensor 1d you just created to change it to a tensor2 with just one value to avoid errors. 

    

    let output = model.predict(newInput);

    output.print();

    return output.squeeze().argMax(); //use the ‘squeeze’ function to convert the 2d output tensor into a 1d tensor.    

  });
  
   answer.array().then(function(index) {

    PREDICTION_ELEMENT.innerText = index;

    PREDICTION_ELEMENT.setAttribute('class', (index === OUTPUTS[OFFSET]) ? 'correct' : 'wrong');

    answer.dispose();

    drawImage(INPUTS[OFFSET]);

  });

}

const CANVAS = document.getElementById('canvas');

const CTX = CANVAS.getContext('2d');


function drawImage(digit) {

  var imageData = CTX.getImageData(0, 0, 28, 28);

  

  for (let i = 0; i < digit.length; i++) {

    imageData.data[i * 4] = digit[i] * 255;      // Red Channel.

    imageData.data[i * 4 + 1] = digit[i] * 255;  // Green Channel.

    imageData.data[i * 4 + 2] = digit[i] * 255;  // Blue Channel.

    imageData.data[i * 4 + 3] = 255;             // Alpha Channel.

  }


  // Render the updated array of data to the canvas itself.

  CTX.putImageData(imageData, 0, 0); 


  // Perform a new classification after a certain interval.

  setTimeout(evaluate, 2000);

}

/*TensorFlow.js also has a tf.browser.toPixels() function
that you can use if you wish to stay completely in Tensor land.
A very useful function if you have all the values in a Tensor of the right shape
for a given image and you want to push those values to a HTML canvas in one line of code.
In this case the above code could be simplified to the following: */

// const CANVAS = document.getElementById('canvas');


// function drawImage(digit) {

//  digit = tf.tensor(digit, [28, 28]);

//  tf.browser.toPixels(digit, CANVAS);

//   // Perform a new classification after a certain interval.

//   setTimeout(evaluate, interval);

// }

function logProgress(epoch, logs) {

  console.log('Data for epoch ' + epoch, Math.sqrt(logs.loss));

}