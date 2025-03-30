const path = require('path');

module.exports = {
  entries: {
    entry: './data/src/entry.js',
    personalDataTypes: './data/src/personalDataTypes.js'
  },
  locales: ['en', 'fr'],
  defaultLocale: 'en', // optional
  createIndex: true,
  localeDir: './data/locale',
  genDir: './data/gen',
  baseDir: path.join(__dirname, '.')
};
