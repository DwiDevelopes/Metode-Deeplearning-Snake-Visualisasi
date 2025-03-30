process.env.NODE_ENV = 'test';

const assert = require('assert');
const setValue = require('../src/set-value');

describe('Set Value Test Suite', () => {
  it('should set a property', done => {
    let obj = {};
    setValue(obj, 'prop', 'value');
    assert.equal(obj.prop, 'value');
    done();
  });

  it('should set a sub prop', done => {
    let obj = {};
    setValue(obj, 'prop.sub', 'value');
    assert.equal(obj.prop.sub, 'value');
    done();
  });

  it('should set an array literal value', done => {
    let obj = {};
    setValue(obj, 'arr[0]', 'value');
    assert.equal(obj.arr[0], 'value');
    done();
  });

  it('should set an array object value', done => {
    let obj = {};
    setValue(obj, 'arr[0].val', 'value');
    assert.equal(obj.arr[0].val, 'value');
    done();
  });

  it('should set a root array item', done => {
    let arr = [];
    setValue(arr, '[0]', 'value');
    assert.equal(arr[0], 'value');
    done();
  });

  it('should throw error when setting an array value to an object', done => {
    let obj = {};
    assert.throws(() => {
      setValue(obj, '[0]', 'value');
    });
    done();
  });

  it('should set imbricated arrays values', done => {
    let obj = {};
    setValue(obj, 'nested[0][0]', 'value');
    assert.equal(obj.nested[0][0], 'value');
    done();
  });

  it('should set imbricated arrays object values', done => {
    let obj = {};
    setValue(obj, 'nested[0][0].val', 'value');
    assert.equal(obj.nested[0][0].val, 'value');
    done();
  });
});
