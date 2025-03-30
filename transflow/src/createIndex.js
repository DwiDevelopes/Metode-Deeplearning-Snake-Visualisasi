const path = require('path');
const fse = require('fs-extra');
module.exports = function createIndex(genDir, locales, entries){
  const indexFilename = path.join(genDir, 'index.js');
  let localesLines = [];
  locales.forEach(locale => {
    let entriesLines = [];
    Object.keys(entries).forEach(entry => {
      entriesLines.push(`    ${entry}: require('./${locale}/${entry}.json')`);
    });
    localesLines.push(`  ${locale}: {\n${entriesLines.join(',\n')}\n  }`);
  });
  let indexContent = `module.exports = {\n${localesLines.join(',\n')}\n};\n`;
  fse.writeFileSync(indexFilename, indexContent, 'utf-8');
};
