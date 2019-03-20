#!/usr/bin/node

const assert = require('assert');
const brain = require('brain.js');
const readline = require('readline');
const fs = require('fs');

let trainingData = [];

let TRAINING_DATA_LEN = 800;
let MAX_ITERATIONS = 1500;

let titledescriptions = JSON.parse(fs.readFileSync('newvies/movies.json', 'utf-8'));

for(let i in titledescriptions) {
	if(TRAINING_DATA_LEN >= 0) {
		let plot = titledescriptions[i].plot;
		if(plot.length > 5) { // chop off the N/A etc
			trainingData.push({ input: titledescriptions[i].title, output: plot});
			//console.log(titledescriptions[i].title, '\n > ', plot);
			TRAINING_DATA_LEN --;
		}
	}
}

console.log('training on: ', trainingData.length, 'sample titles/plots');

// const inputMap = ['0', '+', '*', '=', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
// inputMap.length === inputSize
// to set * for a char it looks like
// internal = [0,1,0,0,0,0,0,0,0,0,0,0,0,0] (or however many 0's lol)

const net = new brain.recurrent.LSTM();

net.train(trainingData, { iterations: MAX_ITERATIONS, errorThresh: 0.025, log: true, logPeriod: 3 });

const tnetJSON = net.toJSON();
fs.writeFileSync('newvies/newvienet.json', JSON.stringify(tnetJSON));

console.log("\nNET IS DONE TRAINING WOOT\n");
console.log("type in a title to a movie and it will make a plot for it!");

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

rl.prompt('title> ');

rl.on('line', function(data) {
    console.log('\t'+net.run(data));
}).on('close', function() {
    process.exit(0);
});