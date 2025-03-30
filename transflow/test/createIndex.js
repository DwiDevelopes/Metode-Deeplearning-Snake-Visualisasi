process.env.NODE_ENV = 'test';

const fse = require('fs-extra');
const path = require('path');
const assert = require('assert');
const options = require('./transflow.config');
const transflow = require('../src/transflow');

const genDir = path.join(__dirname, 'data/gen');

describe('Transflow Test Suite', () => {
  beforeEach((done) => {
    fse.removeSync(path.join(__dirname, 'data/gen'));
    fse.removeSync(path.join(__dirname, 'data/locale'));
    done();
  });

  it('creates the index file', done => {
    transflow(options);
    assert(fse.pathExistsSync(path.join(genDir, 'index.js')), 'Index file should have been created');
    done();
  });
});
