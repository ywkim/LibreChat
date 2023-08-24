module.exports = {
  testEnvironment: 'node',
  clearMocks: true,
  roots: ['<rootDir>'],
  coverageDirectory: 'coverage',
  setupFiles: ['./test/jestSetup.js'],
  transform: {
    '^.+\\.tsx?$': 'ts-jest',
  },
};
