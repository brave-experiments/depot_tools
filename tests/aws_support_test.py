#!/usr/bin/env vpython3
# Copyright (c) 2026 The Brave Authors. All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Tests for Brave's gclient `dep_type: 'aws'` support.

No network and no `gclient sync` are needed:

* `IntegrationTest` checks the aws dep_type is wired into gclient: the
  `aws_support.py` module sits alongside gclient, and the `gclient.py` /
  `gclient_eval.py` dispatch/schema branches are present.

* `BuildUrlTest` covers the pure `build_url` bucket/object joiner.

* `InstallObjectTest` drives `install_object` against a throwaway localhost
  `http.server`, exercising the same download path that hits
  brave-build-deps-public.s3.brave.com -- verification, extraction, the
  gclient-equivalent sidecars, idempotency, and the sha256/size mismatch errors.

* `SchemaTest` checks that `gclient_eval` accepts a DEPS file declaring
  `dep_type: 'aws'`, and rejects an object missing `size_bytes`.
"""

from __future__ import annotations

import functools
import hashlib
import http.server
import io
import os
import sys
import tarfile
import tempfile
import threading
import unittest
import urllib.error
from unittest import mock

# This test lives in tests/; depot_tools (with aws_support.py) is its parent.
_DEPOT_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Inserted at the FRONT so depot_tools' modules resolve -- `gclient_eval`, plus
# the `gclient` / `git_common` / `third_party` that `aws_support` pulls in. The
# front placement matters under vpython3: its virtualenv ships its own
# `third_party` package which would otherwise shadow depot_tools' `third_party`
# and break `from third_party import colorama`.
sys.path.insert(0, _DEPOT_TOOLS)

import aws_support  # noqa: E402  pylint: disable=wrong-import-position
import gclient_eval  # noqa: E402  pylint: disable=wrong-import-position,import-error


def _serve(directory):
    """Start a localhost HTTP server for `directory`; return (base_url, stop)."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=directory)
    httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address

    def stop():
        httpd.shutdown()
        httpd.server_close()

    return 'http://%s:%d' % (host, port), stop


def _serve_flaky(payload, fail_times, fail_status=503):
    """Serve `payload`, but fail the first `fail_times` GETs with `fail_status`.

    Returns (base_url, stop, state); `state['hits']` counts requests received,
    so tests can assert how many attempts `_download` actually made.
    """
    state = {'hits': 0}

    class Handler(http.server.BaseHTTPRequestHandler):

        def do_GET(self):  # noqa: N802 - stdlib callback name
            state['hits'] += 1
            if state['hits'] <= fail_times:
                self.send_error(fail_status)
                return
            self.send_response(200)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            """Silence the per-request stderr logging."""

    httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address

    def stop():
        httpd.shutdown()
        httpd.server_close()

    return 'http://%s:%d' % (host, port), stop, state


def _make_tar_gz(path, payload_name='hello.txt', payload=b'hello aws\n'):
    """Write a .tar.gz containing one file; return (sha256_hex, size_bytes)."""
    with tarfile.open(path, 'w:gz') as tar:
        info = tarfile.TarInfo(payload_name)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        digest.update(f.read())
    return digest.hexdigest(), os.path.getsize(path)


class IntegrationTest(unittest.TestCase):
    """The aws dep_type must be wired into gclient: the module present and the
    dispatch/schema branches in place. A failure means the fork's gclient.py /
    gclient_eval.py edits were lost (e.g. dropped in an upstream merge)."""

    def test_aws_support_module_is_in_place(self):
        self.assertTrue(
            os.path.exists(os.path.join(_DEPOT_TOOLS, 'aws_support.py')),
            'aws_support.py is missing from depot_tools.')

    def test_gclient_dispatch_branch_is_in_place(self):
        with open(os.path.join(_DEPOT_TOOLS, 'gclient.py')) as f:
            # assertTrue rather than assertIn: a miss must not dump the whole
            # file into the failure message.
            self.assertTrue("elif dep_type == 'aws':" in f.read(),
                            'gclient.py is missing the aws dispatch branch.')

    def test_gclient_eval_schema_branch_is_in_place(self):
        with open(os.path.join(_DEPOT_TOOLS, 'gclient_eval.py')) as f:
            self.assertTrue(
                '# AWS content (Brave)' in f.read(),
                'gclient_eval.py is missing the aws schema branch.')


class BuildUrlTest(unittest.TestCase):

    def test_assumes_https_for_a_bare_host(self):
        self.assertEqual(
            aws_support.build_url('bucket.s3.brave.com', 'tool/x-1.tar.gz'),
            'https://bucket.s3.brave.com/tool/x-1.tar.gz')

    def test_honours_an_explicit_scheme_and_trims_slashes(self):
        self.assertEqual(
            aws_support.build_url('http://127.0.0.1:8080/', '/obj.zip'),
            'http://127.0.0.1:8080/obj.zip')


class InstallObjectTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

        self.object_name = 'toolchain/pkg-1.tar.gz'
        bucket = os.path.join(self.tmp, 'bucket')
        os.makedirs(os.path.join(bucket, 'toolchain'))
        self.sha, self.size = _make_tar_gz(
            os.path.join(bucket, self.object_name))

        base_url, stop = _serve(bucket)
        self.addCleanup(stop)
        self.url = aws_support.build_url(base_url, self.object_name)

    def test_downloads_extracts_and_writes_sidecars(self):
        dest = os.path.join(self.tmp, 'out')
        self.assertTrue(
            aws_support.install_object(self.url, self.sha, self.size, dest,
                                       self.object_name))
        self.assertTrue(os.path.exists(os.path.join(dest, 'hello.txt')))
        prefix = self.object_name.replace('/', '_').replace('.', '_')
        self.assertTrue(os.path.exists(os.path.join(dest, '.%s_hash' % prefix)))
        self.assertTrue(
            os.path.exists(os.path.join(dest, '.%s_content_names' % prefix)))

    def test_reinstall_is_a_noop(self):
        dest = os.path.join(self.tmp, 'out')
        self.assertTrue(
            aws_support.install_object(self.url, self.sha, self.size, dest,
                                       self.object_name))
        self.assertFalse(
            aws_support.install_object(self.url, self.sha, self.size, dest,
                                       self.object_name))

    def test_rejects_a_sha256_mismatch(self):
        dest = os.path.join(self.tmp, 'bad-sha')
        with self.assertRaisesRegex(Exception, 'sha256 mismatch'):
            aws_support.install_object(self.url, 'badf00d', self.size, dest,
                                       self.object_name)

    def test_rejects_a_size_mismatch(self):
        dest = os.path.join(self.tmp, 'bad-size')
        with self.assertRaisesRegex(Exception, 'size mismatch'):
            aws_support.install_object(self.url, self.sha, self.size + 1, dest,
                                       self.object_name)


class SchemaTest(unittest.TestCase):

    def test_accepts_an_aws_dep(self):
        deps = '''
deps = {
  "src/third_party/example": {
    "bucket": "brave-build-deps-public.s3.brave.com",
    "dep_type": "aws",
    "objects": [
      {
        "object_name": "example/thing-123.tar.xz",
        "sha256sum": "deadbeef",
        "size_bytes": 1234,
      },
    ],
  },
}
'''
        dep = gclient_eval.Parse(deps,
                                 'DEPS')['deps']['src/third_party/example']
        self.assertEqual(dep['dep_type'], 'aws')
        self.assertEqual(dep['objects'][0]['object_name'],
                         'example/thing-123.tar.xz')

    def test_rejects_an_object_without_size_bytes(self):
        deps = '''
deps = {
  "src/third_party/example": {
    "bucket": "brave-build-deps-public.s3.brave.com",
    "dep_type": "aws",
    "objects": [ { "object_name": "o", "sha256sum": "deadbeef" } ],
  },
}
'''
        with self.assertRaises(Exception):
            gclient_eval.Parse(deps, 'DEPS')


class DownloadRetryTest(unittest.TestCase):
    """`_download` bounds each socket op with a timeout and retries transient
    failures with backoff -- the resilience gclient's gcs path inherits from
    gsutil, which we lack (bare urllib, no resume)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dest = os.path.join(self._tmp.name, 'artifact')
        # Neutralise the backoff sleeps so the suite stays fast.
        sleep = mock.patch.object(aws_support.time, 'sleep')
        sleep.start()
        self.addCleanup(sleep.stop)

    def test_retries_a_transient_5xx_then_succeeds(self):
        payload = b'recovered\n'
        base, stop, state = _serve_flaky(payload, fail_times=2)
        self.addCleanup(stop)
        aws_support._download(base + '/obj', aws_support.Path(self.dest))
        with open(self.dest, 'rb') as f:
            self.assertEqual(f.read(), payload)
        self.assertEqual(state['hits'], 3)  # two failures, then success

    def test_gives_up_after_max_tries(self):
        base, stop, state = _serve_flaky(b'', fail_times=99)
        self.addCleanup(stop)
        with self.assertRaises(urllib.error.HTTPError):
            aws_support._download(base + '/obj', aws_support.Path(self.dest))
        self.assertEqual(state['hits'], aws_support._MAX_TRIES)

    def test_does_not_retry_a_404(self):
        base, stop, state = _serve_flaky(b'', fail_times=99, fail_status=404)
        self.addCleanup(stop)
        with self.assertRaises(urllib.error.HTTPError):
            aws_support._download(base + '/missing',
                                  aws_support.Path(self.dest))
        self.assertEqual(state['hits'], 1)  # a 404 is not worth retrying

    def test_passes_the_socket_timeout_to_urlopen(self):
        captured = {}

        class _FakeResponse:
            status = 200

            def read(self, *_):
                return b''

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        def _fake_urlopen(url, timeout=None):
            del url
            captured['timeout'] = timeout
            return _FakeResponse()

        with mock.patch.object(aws_support.urllib.request, 'urlopen',
                               _fake_urlopen):
            aws_support._download('http://example.invalid/obj',
                                  aws_support.Path(self.dest))
        self.assertEqual(captured['timeout'], aws_support._SOCKET_TIMEOUT)


class IsTransientTest(unittest.TestCase):

    def _http_error(self, code):
        return urllib.error.HTTPError('http://x/o', code, 'msg', {}, None)

    def test_server_errors_and_throttling_are_transient(self):
        self.assertTrue(aws_support._is_transient(self._http_error(503)))
        self.assertTrue(aws_support._is_transient(self._http_error(500)))
        self.assertTrue(aws_support._is_transient(self._http_error(429)))

    def test_other_http_errors_are_not_transient(self):
        self.assertFalse(aws_support._is_transient(self._http_error(404)))
        self.assertFalse(aws_support._is_transient(self._http_error(403)))

    def test_socket_and_url_errors_are_transient(self):
        self.assertTrue(
            aws_support._is_transient(urllib.error.URLError('reset')))
        self.assertTrue(aws_support._is_transient(TimeoutError()))

    def test_unrelated_errors_are_not_transient(self):
        self.assertFalse(aws_support._is_transient(ValueError('nope')))


if __name__ == '__main__':
    unittest.main()
