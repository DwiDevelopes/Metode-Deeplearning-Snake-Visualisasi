module.exports = class Translation {
  constructor(options) {
    this.options = (typeof options === 'string') ? {default: options} : options || {};
  }

  toPO() {
    const opts = this.options || {};
    let obj = {
      msgid: this.name,
      msgstr: opts.default || ''
    };

    if(opts.comment) {
      obj.comments = {translator: opts.comment};
    }

    return obj;
  }

  isHTML() {
    const opts = this.options || {};
    return opts.html === true;
  }
};
