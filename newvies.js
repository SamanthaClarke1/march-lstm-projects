const assert = require('assert');
const brain = require('brain.js');
const readline = require('readline');
const fs = require('fs');
const imdb = require('imdb-api');

const KEY='4da3e8b8';
const NAME_AMT=3;

let allNames = [];

fs.readFileSync('newvies/data.tsv', 'utf-8').split(/\r?\n/).forEach(function(line){
    let tx = line.split('\t'); tx.shift(); tx.shift();
    let tname = tx.shift();
    let tcountry = tx.shift();

    if(tcountry == 'US' || tx[0] == 'en') {
        if(allNames.length < NAME_AMT) {
            allNames.push(tname);
        }
    }
});

console.log('names found:', allNames.length);
console.log('querying for descriptions, etc ...');

for(let i in allNames) {
    setTimeout(function(i) {
        imdb.search({ name: allNames[i], reqtype: "movie" }, { apiKey: KEY, timeout: 10000 }).then(function(d) {
            console.log(d);
        }).catch(function(e) {
            console.log('FUCKING ERRRRRROR');
            console.log(e);
        });
    }, i*1000, i)
}