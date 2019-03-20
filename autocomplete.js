const assert = require('assert');
const brain = require('brain.js');
const readline = require('readline');
const fs = require('fs');

let trainingData = [];
let MAX_ITERATIONS = 399;
let TRAINING_DATA_LEN_MULT = 0.07;

fs.readFileSync('words.txt', 'utf-8').split(/\r?\n/).forEach(function(line){
    if(line.length > 3 && Math.random() < TRAINING_DATA_LEN_MULT)
        trainingData.push(line);
});

console.log('training on: ', trainingData.length, 'sample words');

// const inputMap = ['0', '+', '*', '=', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
// inputMap.length === inputSize
// to set * for a char it looks like
// internal = [0,1,0,0,0,0,0,0,0,0,0,0,0,0] (or however many 0's lol)

const net = new brain.recurrent.LSTM({hiddenLayers: [50]});

net.train(trainingData, {iterations: MAX_ITERATIONS, errorThresh: 0.2, log: true });

const tnetJSON = net.toJSON();
fs.writeFileSync('autocompletenet.json', JSON.stringify(tnetJSON));

console.log("\nNET IS DONE TRAINING WOOT\n");
console.log("type in part of a word and it will auto complete it to what the network thinks is the english word!");

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

rl.prompt('word> ');

rl.on('line', function(data) {
    console.log('\t'+net.run(data));
}).on('close', function() {
    process.exit(0);
});