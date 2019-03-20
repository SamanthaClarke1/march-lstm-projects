#!/usr/bin/node

const brain = require('brain.js');
const path = require('path');
const fs = require('fs');

let MAX_ITERATIONS = 30;
let MAX_SONGS = 5;

function shuffle(a) {
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

let trainingData = []
let files = fs.readdirSync(path.join(__dirname, 'songs'));
files = shuffle(files);

files.forEach(function (fname) {
	if(--MAX_SONGS > 0) {
		trainingData.push(fs.readFileSync(path.join(__dirname, 'songs', fname), 'utf-8'));
	}
});

//console.log(trainingData[0])
console.log(trainingData.length + " rap songs loaded");

const net = new brain.recurrent.LSTM({hiddenLayers: [30,30],  activation: 'sigmoid'});

net.train(trainingData, {iterations: MAX_ITERATIONS, errorThresh: 0.072, log: true, logPeriod: 1 });

const tnetJSON = net.toJSON();
fs.writeFileSync('rapgodnet.json', JSON.stringify(tnetJSON));

console.log('done training!!!');

let txmple = trainingData[Math.floor(Math.random()*trainingData.length)]
let txmpleRI = (txmple.length-100)*Math.random();
let seed = txmple.slice(txmpleRI, txmpleRI+100)
console.log("seed: " + seed);
setTimeout(function pred(net, forecast) {
	out = (net.run(forecast));
	console.log(out);
	setTimeout(pred, 1000, net, out);
}, 1000, net, seed);