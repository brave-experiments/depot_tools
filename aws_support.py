# Copyright (c) 2026 The Brave Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Brave `dep_type: 'aws'` support for gclient (proof-of-concept).

This is a gclient extension that allows the use of regular HTTPS buckets as a
source of dependencies, in the same way gclient supports `gcs`. Although named
`aws`, it actually works with any type of HTTPS bucket.

The purpose of this deps type is to leverage the gclient infrastructure and
dependency management when deploying non-gcs artifacts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# These are resolved from depot_tools at runtime: this module is imported lazily
# from gclient._deps_to_objects(), long after gclient/gclient_scm finish loading.
import gclient
import gclient_scm
import gclient_utils

_LOG = logging.getLogger('aws_support')

# Read in chunks so large toolchain archives never sit fully in memory.
_CHUNK = 1024 * 1024

# The socket timeout for each operation.
_SOCKET_TIMEOUT = 120.0

# The retry/backoff parameters for transient failures
# (5xx, 429, connection reset).
_MAX_TRIES = 5
_RETRY_BASE_DELAY = 5.0
_RETRY_DELAY_MULTIPLE = 1.3


def build_url(bucket: str, object_name: str) -> str:
    """Join a bucket and object name into a download URL.

    `bucket` may be a bare host (`https://` is assumed) or a full base URL,
    optionally with a path prefix. The local test harness relies on `http://`
    being honoured when spelled out explicitly.
    """
    base = bucket if bucket.startswith(('http://', 'https://')) \
        else 'https://' + bucket
    return base.rstrip('/') + '/' + object_name.lstrip('/')


def _file_prefix(object_name: str) -> str:
    """The sidecar prefix gclient uses: object name with `/` and `.` as `_`."""
    return object_name.replace('/', '_').replace('.', '_')


def _artifact_path(dest_dir: Path, object_name: str,
                   output_file: str | None) -> Path:
    """Where the downloaded archive lands (mirrors GcsDependency)."""
    return dest_dir / (output_file or '.' + object_name.replace('/', '_'))


def _hash_path(dest_dir: Path, object_name: str) -> Path:
    return dest_dir / ('.%s_hash' % _file_prefix(object_name))


def _content_names_path(dest_dir: Path, object_name: str) -> Path:
    return dest_dir / ('.%s_content_names' % _file_prefix(object_name))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(_CHUNK), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _is_transient(error: Exception) -> bool:
    """Whether a download error is worth retrying.

    Server errors (5xx) and throttling (429) are transient.
    """
    if isinstance(error, urllib.error.HTTPError):
        return error.code >= 500 or error.code == 429
    return isinstance(error, (urllib.error.URLError, TimeoutError))


def _download_once(url: str, dest: Path) -> None:
    """One download attempt: stream `url` to `dest` over HTTP(S).

    No AWS auth (public buckets). `_SOCKET_TIMEOUT` bounds each blocking socket
    operation, so a stalled connection raises rather than hanging forever.
    """
    with urllib.request.urlopen(  # nosec - trusted DEPS host
            url, timeout=_SOCKET_TIMEOUT) as response:
        if getattr(response, 'status', 200) not in (200, None):
            raise Exception('%s returned HTTP %s' % (url, response.status))
        with dest.open('wb') as out:  # truncates, so a retry starts clean
            shutil.copyfileobj(response, out, _CHUNK)


def _download(url: str, dest: Path) -> None:
    """Stream `url` to `dest`, retrying transient failures with backoff.
    """
    _LOG.info('Downloading %s', url)
    delay = _RETRY_BASE_DELAY
    for attempt in range(_MAX_TRIES):
        try:
            _download_once(url, dest)
            return
        except Exception as error:  # pylint: disable=broad-except
            last = attempt == _MAX_TRIES - 1
            if last or not _is_transient(error):
                raise
            _LOG.warning('Download of %s failed (%s); retrying in %.1fs '
                         '(attempt %d/%d)', url, error, delay, attempt + 2,
                         _MAX_TRIES)
            time.sleep(delay)
            delay *= _RETRY_DELAY_MULTIPLE


def _validate_tar(tar: tarfile.TarFile, prefixes: set) -> bool:
    """Reject members that escape the extraction root (copied from gclient)."""

    def ok(tarinfo):
        if tarinfo.issym() or tarinfo.islnk():
            if os.path.isabs(tarinfo.linkname):
                return False
            target = os.path.normpath(
                os.path.join(os.path.dirname(tarinfo.name), tarinfo.linkname))
            if not any(target.startswith(p) for p in prefixes):
                return False
        if tarinfo.name == '.':
            return True
        cleaned = tarinfo.name
        if cleaned.startswith('./') and len(cleaned) > 2:
            cleaned = cleaned[2:]
        if '../' in cleaned or '..\\' in cleaned or not any(
                cleaned.startswith(p) for p in prefixes):
            return False
        return True

    return all(map(ok, tar.getmembers()))


def _extract(archive_path: Path, dest_dir: Path) -> list:
    """Extract a tar or zip archive into `dest_dir`; return its member names."""
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, 'r:*') as tar:
            names = tar.getnames()
            cleaned = [
                n[2:] if n.startswith('./') and len(n) > 2 else n for n in names
            ]
            top_level = set(n.split('/')[0] for n in cleaned)
            if not _validate_tar(tar, top_level):
                raise Exception('tarfile contains invalid entries')

            def tar_filter(member, path):
                member.mtime = None
                # The extraction filters exist at runtime (3.12+); pylint's
                # stubs may not know them, hence the disables.
                if sys.version_info < (3, 14):
                    default = tarfile.fully_trusted_filter  # pylint: disable=no-member
                else:
                    default = tarfile.data_filter  # pylint: disable=no-member
                return default(member, path)

            tar.extractall(path=dest_dir, filter=tar_filter)
            return names
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            names = zf.namelist()
            zf.extractall(path=dest_dir)
            return names
    # Not an archive: nothing to extract, the artifact itself is the payload.
    return []


def is_installed(dest_dir,
                 object_name: str,
                 sha256sum: str,
                 output_file: str | None = None) -> bool:
    """True when `dest_dir` already holds this object at `sha256sum`."""
    dest_dir = Path(dest_dir)
    if not _artifact_path(dest_dir, object_name, output_file).exists():
        return False
    hash_file = _hash_path(dest_dir, object_name)
    return hash_file.exists() and hash_file.read_text().rstrip() == sha256sum


def install_object(url: str,
                   sha256sum: str,
                   size_bytes: int,
                   dest_dir,
                   object_name: str,
                   output_file: str | None = None) -> bool:
    """Download, verify and extract one object; write the gclient sidecars.

    Returns True when something was downloaded, False when `_hash` already
    recorded `sha256sum` (nothing to do). Raises on a sha256 or size mismatch.
    """
    dest_dir = Path(dest_dir)
    if is_installed(dest_dir, object_name, sha256sum, output_file):
        return False

    artifact = _artifact_path(dest_dir, object_name, output_file)
    for stale in (_hash_path(dest_dir, object_name),
                  _content_names_path(dest_dir, object_name), artifact):
        stale.unlink(missing_ok=True)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        download_path = Path(tmp) / object_name.replace('/', '_')
        _download(url, download_path)

        actual = _sha256(download_path)
        if actual != sha256sum:
            raise Exception('sha256 mismatch for %s: expected %s, got %s' %
                            (url, sha256sum, actual))
        actual_size = download_path.stat().st_size
        if actual_size != size_bytes:
            raise Exception('size mismatch for %s: expected %d, got %d' %
                            (url, size_bytes, actual_size))

        # Place the artifact next to its sidecars (as gclient does for gcs),
        # then extract. Move before extract so a partial extract still leaves a
        # verifiable artifact + matching hash for the next run to redo.
        shutil.move(download_path, artifact)
        names = _extract(artifact, dest_dir)

    _content_names_path(dest_dir,
                        object_name).write_text(json.dumps(names) + '\n')
    _hash_path(dest_dir, object_name).write_text(sha256sum + '\n')
    return True


def parse_aws_dep(parent, name, dep_value, condition, should_process,
                  use_relative_paths, should_process_fn):
    """Expand one `dep_type: 'aws'` DEPS entry into AwsDependency objects.

    Called from the gclient dispatch (added by the patch). Mirrors the `gcs`
    branch:
    one dependency per object, per-object conditions AND-ed with the dep
    condition, and a stale-extraction wipe when any object needs a new download.
    """
    bucket = dep_value['bucket']
    deps = []
    for obj in dep_value['objects']:
        merged = gclient_utils.merge_conditions(condition, obj.get('condition'))
        deps.append(
            AwsDependency(parent=parent,
                          name=name,
                          bucket=bucket,
                          object_name=obj['object_name'],
                          sha256sum=obj['sha256sum'],
                          size_bytes=obj['size_bytes'],
                          output_file=obj.get('output_file'),
                          custom_vars=parent.custom_vars,
                          should_process=should_process
                          and should_process_fn(merged),
                          relative=use_relative_paths,
                          condition=merged))

    if deps and any(d.IsDownloadNeeded()
                    for d in deps) and os.path.exists(deps[0].output_dir):
        # Objects in one dep share an output_dir; we cannot tell which old files
        # to drop, so clear the whole tree before re-extracting (as gcs does).
        _LOG.warning('AWS dependency %s changed; removing old extraction.',
                     name)
        gclient_utils.rmtree(deps[0].output_dir)
    return deps


class AwsDependency(gclient.Dependency):
    """A single object from a Brave AWS bucket, downloaded over HTTPS."""

    def __init__(self, parent, name, bucket, object_name, sha256sum, size_bytes,
                 output_file, custom_vars, should_process, relative, condition):
        self.bucket = bucket
        self.object_name = object_name
        self.sha256sum = sha256sum
        self.size_bytes = size_bytes
        self.output_file = output_file
        super().__init__(parent=parent,
                         name='%s:%s' % (name, object_name),
                         url=build_url(bucket, object_name),
                         managed=None,
                         custom_deps=None,
                         custom_vars=custom_vars,
                         custom_hooks=None,
                         deps_file=None,
                         should_process=should_process,
                         should_recurse=False,
                         relative=relative,
                         condition=condition)

    @property
    def output_dir(self):
        return os.path.join(self.root.root_dir, self.name.split(':')[0])

    # override
    def verify_validity(self):
        # AWS deps allow duplicate object names within a directory, like gcs.
        return True

    # override
    def run(self, revision_overrides, command, args, work_queue, options,
            patch_refs, target_branches, skip_sync_revisions):
        if command in ('runhooks', 'revinfo'):
            return
        if not self.should_process:
            return
        if install_object(self.url, self.sha256sum, self.size_bytes,
                          self.output_dir, self.object_name, self.output_file):
            _LOG.info('Installed %s into %s', self.object_name, self.output_dir)
        super().run(revision_overrides, command, args, work_queue, options,
                    patch_refs, target_branches, skip_sync_revisions)

    def IsDownloadNeeded(self):
        if not self.should_process:
            return False
        return not is_installed(self.output_dir, self.object_name,
                                self.sha256sum, self.output_file)

    # override
    def GetScmName(self):
        return 'aws'

    # override
    def CreateSCM(self, out_cb=None):
        return AwsWrapper(self.url, self.root.root_dir, self.name, self.outbuf,
                          out_cb)


class AwsWrapper(gclient_scm.SCMWrapper):
    """No-op SCM wrapper: downloads happen in AwsDependency.run (like gcs)."""
    name = 'aws'

    def GetCacheMirror(self):
        return None

    def GetActualRemoteURL(self, options):
        return None

    def DoesRemoteURLMatch(self, options):
        del options
        return True

    def revert(self, options, args, file_list):
        """Does nothing."""

    def diff(self, options, args, file_list):
        """Does nothing."""

    def pack(self, options, args, file_list):
        """Does nothing."""

    def revinfo(self, options, args, file_list):
        """Does nothing."""

    def status(self, options, args, file_list):
        """Does nothing."""

    def update(self, options, args, file_list):
        """Does nothing."""
