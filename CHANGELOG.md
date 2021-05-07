Changelog
=========

## 1.10.0 (May 7, 2021)

- Add `vg.apex_and_opposite()`, which is like `vg.apex()` except it
  returns both extreme points.

## 1.9.0 (Jun. 4, 2020)

- Add `vg.argapex()`, which is like `vg.apex()` except it returns the
  index instead of the point.

## 1.8.0 (Apr. 22, 2020)

- Add `vg.shape.columnize()`, a helper function for functions which may,
  accept and return a stack of points or a single point.
- Add `vg.shape.check_value_any()` for checking an input against multiple
  valid shape.
- Fix `vg.euclidean_distance()` so it will correctly accept a single point
  as the first argument with a stack of points as the second argument.
- Correct many erroneous mentions of `3x1` and `kx1` in the documentation
  and error messages.

## 1.7.0 (Jan. 22, 2020)

- Add `vg.average()`.
- `vg.apex()`: Return a copy instead of a view.
- Improve documentation intro.
- Distribute universal wheels.

## 1.6.1 (Oct. 27, 2019)

- Fix `vg.nearest()` and `vg.farthest()` with `ret_index=False`

## 1.6.0 (Oct. 27, 2019)

- Allow `vg.scale_factor()` to accept a mix of stacked and unstacked inputs.

## 1.5.0 (Oct. 24, 2019)

- Add `vg.nearest()`.
- Add `vg.scale_factor()`.
- Add `vg.orient()`.
- Improve documentation, including docs for `vg.almost_collinear()`.

## 1.4.0 (Oct. 6, 2019)

- `vg.scalar_projection` and `vg.project`: Allow stacked inputs for `onto` when
  `vector` is also stacked.

## 1.3.0 (Sep 30, 2019)

Same as 1.2.2 but republished to reflect correct semver.

## 1.2.2 (Sep 29, 2019)

- Add `vg.euclidean_distance`.
- Fix encoding error during installation on Windows 10.

## 1.2.1 (June 30, 2019)

- Fix `vg.shape.check` for scalars.

## 1.2.0 (May 29, 2019)

- Add `vg.cross` and `vg.dot`.
- Allow `vg.perpendicular` to accept a mix of stacked and unstacked inputs.

## 1.1.0 (May 13, 2019)

- Add `vg.almost_unit_length`
- Add `vg.within`

## 1.0.0 (Apr 4, 2019)

- BREAKING CHANGE: Rename `vg.proj()` -> `vg.project()`.
- BREAKING CHANGE: Rename `vg.sproj()` -> `vg.scalar_projection()`.
- BREAKING CHANGE: Move matrix functions into namespace.
- BREAKING CHANGE: Give second argument to `vg.apex()` a better name
- Add `vg.almost_equal()`.
- Improve documentation.

## 0.6.0 (Apr 3, 2019)

- Add `vg.shape.check()` and `vg.shape.check_value()`

## 0.5.2 (Mar 28, 2019)

- Again, fix concurrent install with numpy by avoiding numpy import during install.

## 0.5.1 (Mar 28, 2019)

- Fix concurrent install with numpy by avoiding numpy import during install.

## 0.5.0 (Mar 28, 2019)

- Add `vg.apex()` and `vg.farthest()`
- Fix documentation typos.
- Update dev dependencies.


## 0.4.0 (Jan 17, 2019)

- Add `vg.perpendicular()` for computing the vector perpendicular to two
  others.
- Add `vg.rotate()` for rotating a point or vector some number of degrees
  (or radians) around an axis.
- Add "before and after" examples to the docs.
- **100% code coverage.**

## 0.3.0 (Nov 16, 2018)

- Add principal-component-analysis functions `principal_components()` and
  `major_axis()`.


## 0.2.0 (Oct 5, 2018)

- Vectorize `angle()` and `unsigned_angle()` and improve documentation and
  test coverage.
- Test the return type for `magnitude()` and improve documentation.
- Present documentation on a single page.
- Other doc improvements.


## 0.1.0 (Oct 2, 2018)

Initial release.
