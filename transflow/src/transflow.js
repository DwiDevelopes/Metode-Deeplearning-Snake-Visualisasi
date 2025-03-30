/**
 *
 * @param {Object} options
 * @param {String[]} options.locales
 * @param {Object} options.entries
 * @param {String} [options.genDir] default is ./gen
 * @param {String} [options.localeDir] default is ./locale
 * @param {String} [options.baseDir] default is process.cwd()
 * @param {String} [options.createIndex] default is true
 * @param {String} [options.prettyPrint] default is true
 */
const transflow = (options) => {
  const getTranslationsFromObject = require('./translations-from-object');
  const updateTranslationsFile = require('./update-translations-file');
  const createIndex = require('./createIndex');
  const po2json = require('./po2json');
  const path = require('path');
  const clone = require('clone');

  let settings = Object.assign({}, transflow.defaults, options);

  const dirs = {
    base: settings.baseDir || process.cwd()
  };

  dirs.locales = path.join(dirs.base, settings.localeDir);
  dirs.gen = path.join(dirs.base, settings.genDir);

  for(let entry in settings.entries) {
    if(settings.entries.hasOwnProperty(entry)) {
      const entryPath = path.join(dirs.base, settings.entries[entry]);
      // eslint-disable-next-line
      const src = clone(require(entryPath));
      const translations = getTranslationsFromObject(src);
      settings.locales.forEach(locale => {
        updateTranslationsFile(translations, `${dirs.locales}/${locale}/${entry}.po`);
        po2json(
          src,
          translations,
          `${dirs.locales}/${locale}/${entry}.po`,
          `${dirs.gen}/${locale}/${entry}.json`,
          settings.prettyPrint
        )
      });
    }
  }
  if(settings.createIndex) {
    createIndex(dirs.gen, settings.locales, settings.entries);
  }
};

const Translation = require('./translation');

transflow.t = (opts) => new Translation(opts);
transflow.defaults = module.exports = {
  entries: {},
  locales: [],
  createIndex: true,
  localeDir: './locale',
  genDir: './gen',
  prettyPrint: true
};

module.exports = transflow;
