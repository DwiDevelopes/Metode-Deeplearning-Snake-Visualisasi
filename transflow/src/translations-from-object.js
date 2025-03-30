const Translation = require('./translation');

/**
 *
 * @param src
 * @return {Translation[]}
 */
module.exports = function getTranslationsFromObject(src) {

  let translations = [];

  function gen(obj, buffer) {

    if(typeof buffer === 'undefined') {
      buffer = '';
    }

    if(obj instanceof Translation) {
      obj.name = buffer;
      translations.push(obj);
    } else if(typeof obj === 'object') {
      let k;
      if(Array.isArray(obj)) {
        obj.forEach((item, index) => {
          gen(item, `${buffer}[${index}]`);
        });
      } else {
        for(k in obj) {
          if(obj.hasOwnProperty(k)) {
            gen(obj[k], buffer ? `${buffer}.${k}` : k);
          }
        }
      }
    }
  }

  gen(src);
  return translations;
};
