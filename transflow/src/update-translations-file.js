const fse = require('fs-extra');
const gettextParser = require('gettext-parser');

module.exports = function updateTranslationsFile(translations, file) {
  fse.ensureFileSync(file);
  const input = fse.readFileSync(file, 'utf-8');
  const po = gettextParser.po.parse(input, 'utf-8');
  let updatedTranslations = Object.assign({}, po.translations[''] || {});

  translations.forEach(translation => {
    if(typeof updatedTranslations[translation.name] === 'undefined') {
      updatedTranslations[translation.name] = translation.toPO()
    }
  });

  po.charset = 'utf-8';
  if(!po.headers) {
    po.headers = {};
  }
  po.headers['content-type'] = 'text/plain; charset=utf-8';
  po.translations[''] = updatedTranslations;

  const compiled = gettextParser.po.compile(po);
  fse.writeFileSync(file, compiled);
}
