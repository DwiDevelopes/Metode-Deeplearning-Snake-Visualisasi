/**
 * Set value in target for the given path
 * Just like 'set-value' package but with array support
 *
 * @param {Object} target an object
 * @param {String} path any.prop.even[0].arrays[2].test
 * @param {*} value anything
 */
module.exports = function setValue(target, path, value) {
  const parts = path.replace('][', '].[').split('.');
  let parent = target;
  while(parts.length > 1) {
    // is it an "array" path
    const bracketIndex = parts[0].indexOf('[');
    const sub = () => parts[1].indexOf('[') === 0 ? [] : {};
    if(bracketIndex !== -1) {
      // yes it is, grab the index
      const index = parseInt(parts[0].substring(bracketIndex + 1, parts[0].length - 1));
      // but it might be an array right now
      if(bracketIndex === 0) {
        if(!Array.isArray(parent)) {
          throw new Error('specified target is not an array');
        }
        if(typeof parent[index] === 'undefined') {
          parent[index] = sub();
        }
        parent = parent[index];
      } else {
        // it's an array in an object
        // get the array prop name
        const propName = parts[0].substring(0, bracketIndex);
        // if the array does not exist in the parent, create it
        if(!Array.isArray(parent[propName])) {
          parent[propName] = [];
        }
        // if the index does not exist...
        if(typeof parent[propName][index] === 'undefined') {
          parent[propName][index] = sub();
        }
        parent = parent[propName][index];
      }
    } else {
      if(typeof parent[parts[0]] === 'undefined') {
        parent[parts[0]] = {};
      }
      parent = parent[parts[0]];
    }
    parts.shift()
  }

  const bracketIndex = parts[0].indexOf('[');
  // last element is an array
  if(bracketIndex !== -1) {
    const index = parts[0].substring(bracketIndex + 1, parts[0].length - 1);
    // last element is a raw array
    if(bracketIndex === 0) {
      if(!Array.isArray(parent)) {
        throw new Error('specified target is not an array');
      }
      parent[index] = value;
    } else {
      const propName = parts[0].substring(0, bracketIndex);
      if(!Array.isArray(parent[propName])) {
        parent[propName] = [];
      }
      parent[propName][index] = value;
    }
  } else {
    parent[parts[0]] = value;
  }
};
