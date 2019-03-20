const assert = require('assert');
const brain = require('brain.js');

const TRAINING_DATA_AMT = 499;
const TEST_DATA_AMT = 10;

function getRandomEquation() {
    let tx = Math.floor(Math.random()*10), ty = Math.floor(Math.random()*10);
    let tsym = (Math.random() < 0.5 ? '+' : '*');
    let answer = eval(tx+tsym+ty);
    return [tx+tsym+ty+'=', answer, tsym];
}

let trainingData = [];
for(let i = 0; i < TRAINING_DATA_AMT; i++) {
    let teq = getRandomEquation();
    trainingData.push(teq[0]+teq[1]);
}
console.log(trainingData);

// const inputMap = ['0', '+', '*', '=', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
// inputMap.length === inputSize
// to set * for a char it looks like
// internal = [0,1,0,0,0,0,0,0,0,0,0,0,0,0] (or however many 0's lol)

const net = new brain.recurrent.LSTM({hiddenLayers: [20]});

net.train(trainingData, {errorThresh: 0.039, log: (stats) => console.log(stats) });

for(let i = 0; i < TEST_DATA_AMT; i++) {
    let teq = getRandomEquation();
    let ta = net.run(teq[0]);
    console.log(teq[0], ta, '  ' + (ta == teq[1] ? 'correct!':'wrong!'));
}