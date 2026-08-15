import * as os from 'os';
import * as path from 'path';
import * as fs from 'fs';

let ext = '.so';
if (os.platform() === 'darwin') {
    ext = '.dylib';
} else if (os.platform() === 'win32') {
    ext = '.dll';
}

let libPath = path.resolve(__dirname, '../../ext/libravdb' + ext);
if (!fs.existsSync(libPath)) {
    libPath = path.resolve(__dirname, '../../../cgo/libravdb' + ext);
}
console.log("Loading from:", libPath);
