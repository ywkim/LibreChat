const jose = require('jose');
const passportCustom = require('passport-custom');
const CustomStrategy = passportCustom.Strategy;
const User = require('../models/User');

const joseLogin = async () =>
  new CustomStrategy(async (req, done) => {
    const authHeader = req.headers.authorization;

    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return done(null, false, { message: 'No auth token' });
    }

    const token = authHeader.split(' ')[1];

    try {
      const secret = new TextEncoder().encode(process.env.JWT_SECRET);
      const { payload } = await jose.jwtVerify(token, secret);

      const user = await User.findById(payload.id);
      if (user) {
        done(null, user);
      } else {
        console.log('JoseJwtStrategy => no user found');
        done(null, false, { message: 'No user found' });
      }
    } catch (err) {
      if (err?.code === 'ERR_JWT_EXPIRED') {
        console.error('JoseJwtStrategy => token expired');
      } else {
        console.error('JoseJwtStrategy => error');
        console.error(err);
      }
      done(null, false, { message: 'Invalid token' });
    }
  });

module.exports = joseLogin;
