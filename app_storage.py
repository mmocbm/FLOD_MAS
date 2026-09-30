"""Portable storage for user data, separate from application code and config."""
import ctypes
import filecmp
import os
from pathlib import Path, PureWindowsPath
import shutil
import uuid

APP_FOLDER = 'data files'
CODE_ROOT = Path(__file__).resolve().parent


def documents_folder():
    """Ask Windows for Documents, including redirected/OneDrive locations."""
    if os.name == 'nt':
        from ctypes import wintypes
        class GUID(ctypes.Structure):
            _fields_ = [('data1', wintypes.DWORD), ('data2', wintypes.WORD),
                        ('data3', wintypes.WORD), ('data4', ctypes.c_ubyte * 8)]
        folder_id = GUID.from_buffer_copy(uuid.UUID(
            'FDD39AD0-238F-46AF-ADB4-6C85480369C7').bytes_le)
        shell = ctypes.WinDLL('shell32')
        ole = ctypes.WinDLL('ole32')
        shell.SHGetKnownFolderPath.argtypes = [ctypes.POINTER(GUID), wintypes.DWORD,
                                             wintypes.HANDLE, ctypes.POINTER(ctypes.c_void_p)]
        shell.SHGetKnownFolderPath.restype = ctypes.c_long
        ole.CoTaskMemFree.argtypes = [ctypes.c_void_p]
        ole.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
        ole.CoInitializeEx.restype = ctypes.c_long
        initialized = ole.CoInitializeEx(None, 2)
        address = ctypes.c_void_p()
        try:
            result = shell.SHGetKnownFolderPath(ctypes.byref(folder_id), 0, None,
                                                ctypes.byref(address))
            if result != 0:
                raise OSError(f'Cannot locate Documents folder (HRESULT {result:#x})')
            return Path(ctypes.wstring_at(address.value))
        finally:
            if address.value:
                ole.CoTaskMemFree(address)
            if initialized in (0, 1):
                ole.CoUninitialize()
    # macOS default and Linux XDG Documents support.
    settings = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'user-dirs.dirs'
    if settings.is_file():
        for line in settings.read_text(encoding='utf-8').splitlines():
            if line.startswith('XDG_DOCUMENTS_DIR='):
                value = line.split('=', 1)[1].strip().strip('"')
                return Path(value.replace('$HOME', str(Path.home()))).expanduser()
    return Path.home() / 'Documents'


DATA_ROOT = documents_folder() / APP_FOLDER


def data_path(path):
    """Resolve data below Documents; relocate legacy absolute config paths."""
    text = os.fspath(path).replace('\\', '/')
    relative = Path(text)
    if relative.is_absolute() or PureWindowsPath(text).is_absolute():
        try:
            relative = relative.resolve().relative_to(DATA_ROOT.resolve())
        except ValueError:
            parts = text.split('/')
            anchors = {'files', 'dataset_capture', 'calibration_images',
                       'temp_calibration_images', 'results', 'pattern_data', 'captures'}
            start = next((i for i, part in enumerate(parts) if part.lower() in anchors), None)
            relative = Path(*parts[start:]) if start is not None else Path(parts[-1])
    destination = (DATA_ROOT / relative).resolve()
    if not destination.is_relative_to(DATA_ROOT.resolve()):
        raise ValueError('Saved data paths must stay inside the application Documents folder')
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def initialize_storage(code_root=CODE_ROOT):
    """Move legacy data once, including archives, into Documents/data files."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    marker = DATA_ROOT / '.storage_migrated_v2'
    if marker.exists():
        return
    migrate_storage(code_root)
    marker.touch()


def migration_sources(code_root=CODE_ROOT):
    """List only runtime data; never include code, models or config.json."""
    code_root = Path(code_root).resolve()
    previous = DATA_ROOT.parent / 'MAS Unichela'
    if previous.is_dir() and previous.resolve() != DATA_ROOT.resolve():
        for source in previous.rglob('*'):
            if source.is_file() and not source.is_symlink():
                yield source, source.relative_to(previous), previous, 'previous_documents'
    directories = ('Files', 'calibration_images', 'pattern_data', 'Dataset_capture',
                   'Dataset_Capture', 'temp_calibration_images', 'results', 'captures')
    seen = set()
    for name in directories:
        directory = code_root / name
        if not directory.is_dir():
            continue
        for source in directory.rglob('*'):
            resolved = source.resolve()
            if not source.is_file() or source.is_symlink() or resolved in seen:
                continue
            seen.add(resolved)
            relative = source.relative_to(code_root)
            if name.lower() == 'dataset_capture':
                relative = Path('Dataset_capture', *relative.parts[1:])
            yield source, relative, code_root, 'project'
    # Standalone helpers' saved images and input captures live beside their code.
    for name in ('', 'measure', 'get_mesurement', 'unet', 'Image_Processing', 'CalibrateAPP'):
        directory = code_root / name
        if not directory.is_dir():
            continue
        for source in directory.iterdir():
            if (source.is_file() and not source.is_symlink()
                    and source.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
                    and source.name != 'theme_preview.png'):
                yield source, source.relative_to(code_root), code_root, 'project'


def migrate_storage(code_root=CODE_ROOT):
    """Move data safely; different files with the same name are both preserved."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    moved = 0
    sources = list(migration_sources(code_root))
    cleanup_dirs = set()
    for source, relative, allowed_source, origin in sources:
        # Check resolved boundaries before any move/delete, including symlink parents.
        if not source.resolve().is_relative_to(allowed_source.resolve()):
            raise ValueError(f'Migration source escapes its folder: {source}')
        destination = data_path(relative)
        if not destination.resolve().is_relative_to(DATA_ROOT.resolve()):
            raise ValueError(f'Migration destination escapes Documents data folder: {destination}')
        parent = source.parent
        while parent != allowed_source and parent.is_relative_to(allowed_source):
            cleanup_dirs.add(parent)
            parent = parent.parent
        if destination.exists():
            if filecmp.cmp(source, destination, shallow=False):
                source.unlink()
                moved += 1
                continue
            destination = data_path(Path('.migration_conflicts') / origin / relative)
            suffix = 1
            original = destination
            while destination.exists():
                destination = original.with_name(f'{original.stem}_{suffix}{original.suffix}')
                suffix += 1
        shutil.move(str(source), str(destination))
        moved += 1
    # Remove only empty source directories, never a tree containing code/models.
    for directory in sorted(cleanup_dirs, key=lambda p: len(p.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass
    previous = DATA_ROOT.parent / 'MAS Unichela'
    if previous.is_dir():
        try:
            previous.rmdir()
        except OSError:
            pass
    return moved


if __name__ == '__main__':
    count = migrate_storage()
    (DATA_ROOT / '.storage_migrated_v2').touch()
    print(f'Moved {count} data files into {DATA_ROOT}')
