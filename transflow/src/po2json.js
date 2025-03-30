const fse = require('fs-extra');
const path = require('path');
const gettextParser = require('gettext-parser');
const setValue = require('./set-value');

module.exports = function po2json(srcObject, translations, file, output, prettyPrint) {
  fse.ensureFileSync(file);
  fse.ensureFileSync(output);

  const input = fse.readFileSync(file, 'utf-8');
  const po = gettextParser.po.parse(input, 'utf-8');
  const poTranslations = po.translations[''] || {};

  let localizedObject = srcObject;

  translations.forEach(translation => {
    if(translation.isHTML() === true) {
      const htmlFile = `${path.dirname(file)}/html/${translation.name}.html`;
      fse.ensureFileSync(htmlFile);
      setValue(localizedObject, translation.name, fse.readFileSync(htmlFile, 'utf-8'));
    } else if(poTranslations[translation.name]) {
      setValue(localizedObject, translation.name, poTranslations[translation.name].msgstr[0]);
    }
  });
  fse.writeFileSync(output, JSON.stringify(localizedObject, null, prettyPrint ? 2 : null));
};

