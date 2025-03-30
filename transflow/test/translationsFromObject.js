process.env.NODE_ENV = 'test';

const assert = require('assert');
const translationsFromObject = require('../src/translations-from-object');
const entry = require('./data/src/entry');

describe('Translations from Object Test Suite', () => {
  const translations = translationsFromObject(entry);
  const translationByName = (name) => translations.filter(t => t.name === name)[0];

  it('should generate translations from object', done => {
    assert(Array.isArray(translations), 'Translations have to be an array');
    done();
  });
  it('should handle simple props', done => {
    assert.equal(translations[0].name, 'simpleProp', 'Translations have to be an array');
    done();
  });
  it('should handle auto default props', done => {
    assert.equal(
      translationByName('autoDefaultProp').options.default,
      'autoDefault',
      'Translations have to be an array'
    );
    done();
  });
});
